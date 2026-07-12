import {
  VolumeWeightedKAMAIndicator,
  volumeWeightedKamaWarmupSamples,
} from "./indicators.js";
import {
  perfectMarginOracle,
  type PerfectMarginOracleResult,
} from "./perfect-margin-oracle.js";
import type {
  BacktestOraclePath,
  BacktestOraclePoint,
  BacktestOracleState,
} from "./backtest-trace.js";
import type {
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  Candle,
} from "./legacy/types.js";
import type { TradingApi, TradingCandle } from "./trading-api.js";
import { signalBeyondFriction } from "./signal-memory.js";
import { KamaRateNoise } from "./kama-rate-noise.js";

const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
const F1_WEIGHT = 0.6;
const AGREEMENT_WEIGHT = 0.3;
const CLEANLINESS_WEIGHT = 0.1;

export type VwKamaDeadbandMode = "flat" | "hold";
export type VwKamaThresholdMode = "static" | "adaptive";

export interface VwKamaParameters {
  efficiencyMs: number;
  fastMs: number;
  slowMs: number;
  power: number;
  volumeMs: number;
  volumeCap: number;
  volumePower: number;
  deadbandBpsHour: number;
  deadbandMode: VwKamaDeadbandMode;
  thresholdMode?: VwKamaThresholdMode;
  thresholdLookbackMs?: number;
  thresholdNoiseMultiplier?: number;
}

export interface VwKamaInspectorWindow {
  id: string;
  label: string;
  group: string;
  startTime: number;
  endTime: number;
}

export interface VwKamaInspectorRequest {
  windowId: string;
  intervalMs: number;
  parameters: VwKamaParameters;
  oracleFriction: number;
  matchWindowMs: number;
  timingHalfLifeMs: number;
  warmupMultiple: number;
}

export interface VwKamaCandleRangeRequest extends VwKamaInspectorRequest {
  startTime: number;
  endTime: number;
  maxCandles?: number;
}

export interface VwKamaCandleRangeResponse {
  windowId: string;
  intervalMs: number;
  renderIntervalMs: number;
  startTime: number;
  endTime: number;
  sourceCandleCount: number;
  candles: Candle[];
  kamaSeries: BacktestChartSmaSeries;
}

export interface VwKamaInspectorCatalog {
  windows: VwKamaInspectorWindow[];
  scales: Array<{ label: string; intervalMs: number }>;
  defaults: VwKamaInspectorRequest;
  presets: VwKamaPreset[];
}

export interface VwKamaPreset {
  id: string;
  label: string;
  scope: "global" | "window";
  windowId: string | null;
  intervalMs?: number | null;
  parameters: VwKamaParameters;
  score?: number;
  source?: string;
}

export interface VwKamaTransition {
  time: number;
  price: number;
  side: "buy" | "sell";
  fromState: BacktestOracleState;
  state: BacktestOracleState;
  matchedTime: number | null;
  lagMs: number | null;
  timingCredit: number;
}

export interface VwKamaAccuracyMetrics {
  score: number;
  precision: number;
  recall: number;
  f1: number;
  rawPrecision: number;
  rawRecall: number;
  exposureAgreement: number;
  noiseSignalRatio: number | null;
  signalCleanliness: number;
  signalsPerDay: number;
  signalCount: number;
  oracleCount: number;
  matchedCount: number;
  extraSignalCount: number;
  missedOracleCount: number;
  lagP50Ms: number | null;
  lagP90Ms: number | null;
  lagP95Ms: number | null;
  lagMedianSignedMs: number | null;
}

export interface VwKamaStatePoint {
  time: number;
  candidate: BacktestOracleState;
  oracle: BacktestOracleState;
}

export interface VwKamaInspectorResponse {
  window: VwKamaInspectorWindow;
  intervalMs: number;
  renderIntervalMs: number;
  candleCount: number;
  sourceSegmentCount: number;
  scoredSegmentCount: number;
  elapsedMs: number;
  metrics: VwKamaAccuracyMetrics;
  candles: Candle[];
  kamaSeries: BacktestChartSmaSeries;
  annotations: BacktestChartAnnotation[];
  oracle: BacktestOraclePath;
  candidatePath: { points: BacktestOraclePoint[] };
  statePoints: VwKamaStatePoint[];
  candidateTransitions: VwKamaTransition[];
  oracleTransitions: VwKamaTransition[];
}

export interface EvaluateVwKamaOptions extends Omit<VwKamaInspectorRequest, "windowId"> {
  scoreStartTime: number;
  maxPoints?: number;
  oracleResult?: PerfectMarginOracleResult;
  includeTrace?: boolean;
}

export type VwKamaEvaluation = Omit<
  VwKamaInspectorResponse,
  | "window"
  | "renderIntervalMs"
  | "sourceSegmentCount"
  | "scoredSegmentCount"
  | "elapsedMs"
  | "candles"
>;

export function evaluateVwKamaOracle(
  candles: readonly TradingCandle[],
  options: EvaluateVwKamaOptions,
): VwKamaEvaluation {
  if (candles.length === 0) throw new Error("VW-KAMA inspection requires candles.");
  validate(options);
  const scoreStart = lowerBound(candles, options.scoreStartTime);
  if (scoreStart >= candles.length) throw new Error("VW-KAMA score window has no candles.");

  const periods = {
    efficiencyPeriod: samples(options.parameters.efficiencyMs, options.intervalMs),
    fastPeriod: samples(options.parameters.fastMs, options.intervalMs),
    slowPeriod: samples(options.parameters.slowMs, options.intervalMs),
    volumePeriod: samples(options.parameters.volumeMs, options.intervalMs),
  };
  const warmup = volumeWeightedKamaWarmupSamples(periods, options.warmupMultiple);
  const thresholdSamples = samples(options.parameters.thresholdLookbackMs ?? options.intervalMs, options.intervalMs);
  const feedStart = Math.max(0, scoreStart - Math.max(warmup, thresholdSamples * options.warmupMultiple));
  const indicator = new VolumeWeightedKAMAIndicator({} as TradingApi, {
    ...periods,
    power: options.parameters.power,
    volumeCap: options.parameters.volumeCap,
    volumePower: options.parameters.volumePower,
  });
  const candidateStates = new Int8Array(candles.length);
  const candidateTransitions: VwKamaTransition[] = [];
  const points: BacktestOraclePoint[] = [];
  const kamaSeries: BacktestChartSmaSeries = {
    index: -1,
    windowSec: options.parameters.slowMs / 1_000,
    label: "VW-KAMA",
    color: "#f472b6",
    points: [],
  };
  const maxPoints = Math.max(1, Math.round(options.maxPoints ?? 2_000));
  const sampleEvery = Math.max(1, Math.ceil((candles.length - scoreStart) / maxPoints));
  const includeTrace = options.includeTrace !== false;
  let current = 0;
  let lastSignalPrice: number | null = null;
  const rateNoise = new KamaRateNoise(thresholdSamples);

  for (let index = feedStart; index < candles.length; index += 1) {
    const candle = candles[index]!;
    indicator.onTick({ eventTime: candle.closeTime, candle });
    const kama = indicator.indicator();
    const rate = kama > 0
      ? indicator.derivative() / kama * 10_000 * HOUR_MS / options.intervalMs
      : 0;
    const noise = rateNoise.update(rate);
    const threshold = options.parameters.deadbandBpsHour
      + (options.parameters.thresholdMode === "adaptive"
        ? noise * Math.max(0, options.parameters.thresholdNoiseMultiplier ?? 0)
        : 0);
    const thresholded = rate > threshold ? 1 : rate < -threshold ? -1 : 0;
    const desired = options.parameters.deadbandMode === "hold" && thresholded === 0
      ? current : thresholded;
    if (
      desired !== current
      && signalBeyondFriction(candle.close, lastSignalPrice, options.oracleFriction)
    ) {
      if (index >= scoreStart) {
        candidateTransitions.push(baseTransition(candle, current, desired));
        if (includeTrace) points.push(statePoint(candle, current, desired));
      }
      current = desired;
      lastSignalPrice = candle.close;
    }
    candidateStates[index] = current;
    if (includeTrace && index >= scoreStart && ((index - scoreStart) % sampleEvery === 0 || index === candles.length - 1)) {
      kamaSeries.points.push({ time: candle.closeTime, value: kama });
      if (points.at(-1)?.time !== candle.closeTime) points.push(statePoint(candle, current, current));
    }
  }

  const oracleResult = options.oracleResult ?? perfectMarginOracle(candles, {
    startingQuote: 1,
    leverage: 1,
    friction: options.oracleFriction,
    eventMode: "close",
    maxPathCandles: maxPoints,
  });
  const oracleStates = new Int8Array(candles.length);
  for (let index = 0; index < oracleStates.length; index += 1) {
    oracleStates[index] = exposureFromCode(oracleResult.stateCodes[index] ?? 0);
  }
  const oracleTransitions: VwKamaTransition[] = [];
  for (let index = Math.max(1, scoreStart); index < candles.length; index += 1) {
    const previous = oracleStates[index - 1] ?? 0;
    const current = oracleStates[index] ?? 0;
    if (current !== previous) oracleTransitions.push(baseTransition(candles[index]!, previous, current));
  }

  const alignment = alignVwKamaTransitions(candidateTransitions, oracleTransitions, options);
  let stateCredit = 0;
  for (let index = scoreStart; index < candles.length; index += 1) {
    stateCredit += 1 - Math.abs(candidateStates[index]! - oracleStates[index]!) / 2;
  }
  const stateCount = candles.length - scoreStart;
  const precision = eventRatio(alignment.credit, candidateTransitions.length, oracleTransitions.length);
  const recall = eventRatio(alignment.credit, oracleTransitions.length, candidateTransitions.length);
  const f1 = harmonic(precision, recall);
  const matchedCount = alignment.matches.length;
  const extraSignalCount = candidateTransitions.length - matchedCount;
  const signalCleanliness = candidateTransitions.length > 0
    ? matchedCount / candidateTransitions.length
    : 1;
  const lags = alignment.matches.map((item) => item.lagMs);
  const absoluteLags = lags.map(Math.abs);
  const metrics: VwKamaAccuracyMetrics = {
    score: vwKamaScore(f1, stateCredit / stateCount, signalCleanliness),
    precision,
    recall,
    f1,
    rawPrecision: eventRatio(alignment.matches.length, candidateTransitions.length, oracleTransitions.length),
    rawRecall: eventRatio(alignment.matches.length, oracleTransitions.length, candidateTransitions.length),
    exposureAgreement: stateCredit / stateCount,
    noiseSignalRatio: noiseSignalRatio(extraSignalCount, matchedCount),
    signalCleanliness,
    signalsPerDay: candidateTransitions.length / Math.max(options.intervalMs / DAY_MS, stateCount * options.intervalMs / DAY_MS),
    signalCount: candidateTransitions.length,
    oracleCount: oracleTransitions.length,
    matchedCount,
    extraSignalCount,
    missedOracleCount: oracleTransitions.length - matchedCount,
    lagP50Ms: percentile(absoluteLags, 0.5),
    lagP90Ms: percentile(absoluteLags, 0.9),
    lagP95Ms: percentile(absoluteLags, 0.95),
    lagMedianSignedMs: percentile(lags, 0.5),
  };
  const scoredStartTime = candles[scoreStart]!.closeTime;
  const oraclePoints = includeTrace ? slicePath(oracleResult.path.points, scoredStartTime) : [];
  const sampledIndexes = new Set(kamaSeries.points.map((point) => point.time));
  const statePoints: VwKamaStatePoint[] = [];
  if (includeTrace) {
    for (let index = scoreStart; index < candles.length; index += 1) {
      const candle = candles[index]!;
      if (!sampledIndexes.has(candle.closeTime)) continue;
      statePoints.push({
        time: candle.closeTime,
        candidate: stateName(candidateStates[index]!),
        oracle: stateName(oracleStates[index]!),
      });
    }
  }

  return {
    intervalMs: options.intervalMs,
    candleCount: stateCount,
    metrics,
    kamaSeries,
    annotations: includeTrace ? candidateTransitions.map((item) => ({
      time: item.time,
      price: item.price,
      kind: item.side === "buy" ? "buy-signal" : "sell-signal",
      label: `VW-KAMA ${item.fromState} → ${item.state}`,
      signalState: item.state,
      reason: item.lagMs === null ? "No one-to-one oracle match" : `Oracle timing offset ${item.lagMs} ms`,
    })) : [],
    oracle: { ...oracleResult.path, points: oraclePoints },
    candidatePath: { points: includeTrace ? points.sort((left, right) => left.time - right.time) : [] },
    statePoints,
    candidateTransitions,
    oracleTransitions,
  };
}

export function vwKamaScore(
  f1: number,
  exposureAgreement: number,
  signalCleanliness: number,
): number {
  return f1 * F1_WEIGHT
    + exposureAgreement * AGREEMENT_WEIGHT
    + signalCleanliness * CLEANLINESS_WEIGHT;
}

export function noiseSignalRatio(extra: number, matched: number): number | null {
  return matched > 0 ? extra / matched : extra > 0 ? null : 0;
}

export interface VwKamaTransitionMatch {
  candidateIndex: number;
  oracleIndex: number;
  lagMs: number;
  timingCredit: number;
}

export interface VwKamaTransitionAlignment {
  credit: number;
  matches: VwKamaTransitionMatch[];
}

interface AlignmentNode extends VwKamaTransitionMatch {
  previous: number;
  score: AlignmentScore;
}

interface AlignmentScore {
  credit: number;
  count: number;
  absoluteLagMs: number;
  node: number;
}

const EMPTY_ALIGNMENT: AlignmentScore = {
  credit: 0,
  count: 0,
  absoluteLagMs: 0,
  node: -1,
};

export function alignVwKamaTransitions(
  candidates: VwKamaTransition[],
  oracle: VwKamaTransition[],
  options: Pick<EvaluateVwKamaOptions, "matchWindowMs" | "timingHalfLifeMs">,
): VwKamaTransitionAlignment {
  if (options.matchWindowMs < 0 || options.timingHalfLifeMs <= 0) {
    throw new Error("VW-KAMA alignment window must be non-negative and half-life positive.");
  }
  if (!chronological(candidates) || !chronological(oracle)) {
    throw new Error("VW-KAMA transitions must be chronological.");
  }
  resetMatches(candidates);
  resetMatches(oracle);
  const byState: Record<BacktestOracleState, Array<{ index: number; time: number }>> = {
    flat: [],
    long: [],
    short: [],
  };
  for (let index = 0; index < oracle.length; index += 1) {
    const transition = oracle[index]!;
    byState[transition.state].push({ index, time: transition.time });
  }
  const tree: Array<AlignmentScore | undefined> = Array(oracle.length + 1);
  const nodes: AlignmentNode[] = [];
  for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
    const candidate = candidates[candidateIndex]!;
    const targets = byState[candidate.state];
    const first = timeLowerBound(targets, candidate.time - options.matchWindowMs);
    const end = timeUpperBound(targets, candidate.time + options.matchWindowMs);
    const pending: Array<Omit<AlignmentNode, "previous" | "score"> & {
      previous: AlignmentScore;
      score: Omit<AlignmentScore, "node">;
    }> = [];
    for (let targetIndex = first; targetIndex < end; targetIndex += 1) {
      const target = targets[targetIndex]!;
      const lagMs = candidate.time - target.time;
      const timingCredit = Math.exp(-Math.LN2 * Math.abs(lagMs) / options.timingHalfLifeMs);
      const previous = queryAlignment(tree, target.index);
      const score = {
        credit: previous.credit + timingCredit,
        count: previous.count + 1,
        absoluteLagMs: previous.absoluteLagMs + Math.abs(lagMs),
      };
      if (compareAlignment(score, queryAlignment(tree, target.index + 1)) <= 0) continue;
      pending.push({ candidateIndex, oracleIndex: target.index, lagMs, timingCredit, previous, score });
    }
    for (const item of pending) {
      const node = nodes.length;
      const score = { ...item.score, node };
      nodes.push({
        candidateIndex: item.candidateIndex,
        oracleIndex: item.oracleIndex,
        lagMs: item.lagMs,
        timingCredit: item.timingCredit,
        previous: item.previous.node,
        score,
      });
      updateAlignment(tree, item.oracleIndex, score);
    }
  }
  const best = queryAlignment(tree, oracle.length);
  const matches: VwKamaTransitionMatch[] = [];
  for (let node = best.node; node >= 0; node = nodes[node]!.previous) {
    const { candidateIndex, oracleIndex, lagMs, timingCredit } = nodes[node]!;
    matches.push({ candidateIndex, oracleIndex, lagMs, timingCredit });
  }
  matches.reverse();
  for (const match of matches) {
    const candidate = candidates[match.candidateIndex]!;
    const target = oracle[match.oracleIndex]!;
    candidate.matchedTime = target.time;
    candidate.lagMs = match.lagMs;
    candidate.timingCredit = match.timingCredit;
    target.matchedTime = candidate.time;
    target.lagMs = -match.lagMs;
    target.timingCredit = match.timingCredit;
  }
  return { credit: best.credit, matches };
}

function queryAlignment(tree: Array<AlignmentScore | undefined>, end: number): AlignmentScore {
  let best = EMPTY_ALIGNMENT;
  for (let index = end; index > 0; index -= index & -index) {
    const candidate = tree[index];
    if (candidate && compareAlignment(candidate, best) > 0) best = candidate;
  }
  return best;
}

function updateAlignment(
  tree: Array<AlignmentScore | undefined>,
  oracleIndex: number,
  score: AlignmentScore,
): void {
  for (let index = oracleIndex + 1; index < tree.length; index += index & -index) {
    const current = tree[index];
    if (!current || compareAlignment(score, current) > 0) tree[index] = score;
  }
}

function compareAlignment(
  left: Omit<AlignmentScore, "node">,
  right: Omit<AlignmentScore, "node">,
): number {
  const tolerance = Number.EPSILON * 32 * Math.max(1, Math.abs(left.credit), Math.abs(right.credit));
  if (left.credit > right.credit + tolerance) return 1;
  if (right.credit > left.credit + tolerance) return -1;
  if (left.count !== right.count) return left.count > right.count ? 1 : -1;
  if (left.absoluteLagMs !== right.absoluteLagMs) return left.absoluteLagMs < right.absoluteLagMs ? 1 : -1;
  return 0;
}

function resetMatches(transitions: VwKamaTransition[]): void {
  for (const transition of transitions) {
    transition.matchedTime = null;
    transition.lagMs = null;
    transition.timingCredit = 0;
  }
}

function chronological(transitions: VwKamaTransition[]): boolean {
  return transitions.every((transition, index) => index === 0 || transition.time >= transitions[index - 1]!.time);
}

function timeLowerBound(values: Array<{ time: number }>, time: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle]!.time < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function timeUpperBound(values: Array<{ time: number }>, time: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle]!.time <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function baseTransition(candle: TradingCandle, from: number, to: number): VwKamaTransition {
  return {
    time: candle.closeTime,
    price: candle.close,
    side: to - from > 0 ? "buy" : "sell",
    fromState: stateName(from),
    state: stateName(to),
    matchedTime: null,
    lagMs: null,
    timingCredit: 0,
  };
}

function statePoint(candle: TradingCandle, from: number, to: number): BacktestOraclePoint {
  const fromState = stateName(from);
  const state = stateName(to);
  return {
    time: candle.closeTime,
    price: candle.close,
    fromState,
    state,
    action: fromState === state ? "hold" : fromState === "flat" ? "open" : state === "flat" ? "close" : "switch",
  };
}

function slicePath(points: BacktestOraclePoint[], startTime: number): BacktestOraclePoint[] {
  const index = points.findIndex((point) => point.time >= startTime);
  if (index <= 0) return index < 0 ? points.slice(-1) : points.slice();
  return points.slice(index - 1);
}

function exposureFromCode(code: number): -1 | 0 | 1 {
  return code === 1 ? 1 : code === 2 ? -1 : 0;
}

function stateName(exposure: number): BacktestOracleState {
  return exposure > 0 ? "long" : exposure < 0 ? "short" : "flat";
}

function samples(durationMs: number, intervalMs: number): number {
  return Math.max(1, Math.round(durationMs / intervalMs));
}

function lowerBound(candles: readonly TradingCandle[], time: number): number {
  let low = 0;
  let high = candles.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (candles[middle]!.openTime < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function validate(options: EvaluateVwKamaOptions): void {
  const positive = [
    options.intervalMs,
    options.parameters.efficiencyMs,
    options.parameters.fastMs,
    options.parameters.slowMs,
    options.parameters.power,
    options.parameters.volumeMs,
    options.parameters.volumeCap,
    options.timingHalfLifeMs,
    options.warmupMultiple,
  ];
  if (positive.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("VW-KAMA durations and powers must be positive.");
  }
  const nonNegative = [
    options.parameters.volumePower,
    options.parameters.deadbandBpsHour,
    options.parameters.thresholdNoiseMultiplier ?? 0,
    options.oracleFriction,
    options.matchWindowMs,
  ];
  if (nonNegative.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("VW-KAMA thresholds and friction cannot be negative.");
  }
  if (options.parameters.thresholdMode === "adaptive"
    && (!Number.isFinite(options.parameters.thresholdLookbackMs)
      || (options.parameters.thresholdLookbackMs ?? 0) <= 0)) {
    throw new Error("VW-KAMA adaptive threshold lookback must be positive.");
  }
}

function eventRatio(value: number, total: number, opposite: number): number {
  return total > 0 ? value / total : opposite === 0 ? 1 : 0;
}

function harmonic(left: number, right: number): number {
  return left + right > 0 ? 2 * left * right / (left + right) : 0;
}

function percentile(values: number[], quantile: number): number | null {
  if (values.length === 0) return null;
  const sorted = values.slice().sort((left, right) => left - right);
  const index = (sorted.length - 1) * quantile;
  const lower = Math.floor(index);
  const fraction = index - lower;
  return sorted[lower]! * (1 - fraction) + (sorted[lower + 1] ?? sorted[lower]!) * fraction;
}
