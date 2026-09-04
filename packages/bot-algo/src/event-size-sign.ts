import type { EventCandle, MoveAtom, MoveSample } from "./event-distribution.js";
import { eventLeaf, eventMoveLabel, validateEventDistribution, type EventDistribution } from "./event-distribution.js";
import { predictEventSign, trainEventSign, type EventSignHead } from "./event-sign.js";

export const EVENT_FAST_VOLATILITY_INPUTS = ["log1p-range-1m-bps", "log1p-rv-5m-bps", "log1p-rv-15m-bps"] as const;
/** Known at the current candle close; all lookbacks fit inside the existing
 * one-day purge. Head-only inputs do not change the frozen base state map. */
export function eventFastVolatilityFeatures(c: readonly EventCandle[], index: number): number[] {
  if (!Number.isInteger(index) || index < 15 || index >= c.length) throw new Error("Invalid volatility feature index");
  let short = 0, medium = 0;
  for (let i = index - 14; i <= index; i++) {
    if (c[i].openTime - c[i - 1].openTime !== 60_000) throw new Error("Gap in volatility feature history");
    const r = Math.log(c[i].close / c[i - 1].close);
    medium += r * r;
    if (i > index - 5) short += r * r;
  }
  const features = [Math.log1p(Math.log(c[index].high / c[index].low) * 1e4),
    Math.log1p(Math.sqrt(short) * 1e4), Math.log1p(Math.sqrt(medium) * 1e4)];
  if (features.some(v => !Number.isFinite(v) || v < 0)) throw new Error("Invalid volatility feature prices");
  return features;
}

/** Factorizes the active outcome as P(size regime | x) P(sign | regime, x).
 * Training labels may use realized size; inference only accepts past features. */
export interface EventSizeSignHead {
  quantile: number; thresholdLogBps: number; samples: number;
  gate: EventSignHead; ordinarySign: EventSignHead; largeSign: EventSignHead;
}

export function trainEventSizeSign(samples: readonly Pick<MoveSample, "features" | "return">[], penalty: number,
  quantile: number): EventSizeSignHead {
  if (!(quantile > 0 && quantile < 1) || samples.some(s => !(s.return > -1) || !Number.isFinite(s.return)))
    throw new Error("Invalid size/sign targets or quantile");
  const active = samples.filter(s => s.return !== 0);
  const magnitudes = active.map(s => Math.abs(Math.log1p(s.return)) * 1e4).sort((a, b) => a - b);
  const thresholdLogBps = magnitudes[Math.floor(quantile * magnitudes.length)];
  if (!(thresholdLogBps > 0)) throw new Error("No active size/sign targets");
  const large = (s: Pick<MoveSample, "return">) => Math.abs(Math.log1p(s.return)) * 1e4 >= thresholdLogBps;
  const ordinary = active.filter(s => !large(s)), tail = active.filter(large);
  if (ordinary.length < 2 || tail.length < 2) throw new Error("Insufficient outcomes in a size regime");
  return { quantile, thresholdLogBps, samples: active.length,
    gate: trainEventSign(active.map(s => ({ features: s.features, return: large(s) ? 1 : -1 })), penalty),
    ordinarySign: trainEventSign(ordinary, penalty), largeSign: trainEventSign(tail, penalty) };
}

/** Ordinary down/up, large down/up; zero is a separate fifth base-law group. */
export function eventSizeSignGroup(value: number, thresholdLogBps: number): number {
  if (!(thresholdLogBps > 0) || !Number.isFinite(thresholdLogBps) || !(value > -1) || !Number.isFinite(value))
    throw new Error("Invalid size/sign grouping");
  if (value === 0) return 4;
  return (Math.abs(Math.log1p(value)) * 1e4 >= thresholdLogBps ? 2 : 0) + Number(value > 0);
}

export function predictEventSizeSigns(head: EventSizeSignHead, features: readonly number[]): number[] {
  // Each component consumes a declared prefix, allowing event-history inputs
  // on the size gate while retaining the frozen sign heads' original basis.
  const width = Math.max(head.gate.means.length, head.ordinarySign.means.length, head.largeSign.means.length);
  if (features.length !== width) throw new Error("Invalid size/sign feature width");
  const predict = (component: EventSignHead) => predictEventSign(component, features.slice(0, component.means.length));
  const large = predict(head.gate), upOrdinary = predict(head.ordinarySign), upLarge = predict(head.largeSign);
  return [(1 - large) * (1 - upOrdinary), (1 - large) * upOrdinary, large * (1 - upLarge), large * upLarge];
}

export function trainEventSizeGate(samples: readonly Pick<MoveSample, "features" | "return">[], base: EventSizeSignHead,
  penalty: number): EventSizeSignHead {
  if (samples.some(s => !(s.return > -1) || !Number.isFinite(s.return))) throw new Error("Invalid size-gate targets");
  const active = samples.filter(s => s.return !== 0);
  return { ...base, samples: active.length, gate: trainEventSign(active.map(s => ({ features: s.features,
    return: Math.abs(Math.log1p(s.return)) * 1e4 >= base.thresholdLogBps ? 1 : -1 })), penalty) };
}

export function eventSizeSignMass(kernel: readonly MoveAtom[], thresholdLogBps: number): number[] {
  const masses = [0, 0, 0, 0, 0];
  for (const a of kernel) masses[eventSizeSignGroup(a.return, thresholdLogBps)] += a.probability;
  return masses;
}

/** Select which learned factors replace the state law: size gate, ordinary
 * sign, large sign. Intermediate weights blend each conditional probability.
 * Unsupported state regimes retain the full base law in mixEventSizeSigns. */
export function selectEventSizeSignComponents(base: readonly number[], predicted: readonly number[], weights: readonly number[]): number[] {
  mixEventSizeSigns(base, predicted, 1);
  if (weights.length !== 3 || weights.some(w => !Number.isFinite(w) || w < 0 || w > 1)) throw new Error("Invalid size/sign component weights");
  if (weights.every(w => w === 1) || base.slice(0, 4).some(v => !v)) return [...predicted];
  const ordinary = base[0] + base[1], large = base[2] + base[3];
  const predictedOrdinary = predicted[0] + predicted[1], predictedLarge = predicted[2] + predicted[3];
  const blend = (a: number, b: number, w: number) => (1 - w) * a + w * b;
  const gate = blend(large / (ordinary + large), predictedLarge, weights[0]);
  const ordinaryUp = blend(base[1] / ordinary, predicted[1] / predictedOrdinary, weights[1]);
  const largeUp = blend(base[3] / large, predicted[3] / predictedLarge, weights[2]);
  return [(1 - gate) * (1 - ordinaryUp), (1 - gate) * ordinaryUp, gate * (1 - largeUp), gate * largeUp];
}

/** Never invent an unobserved conditional path law. Unsupported regimes retain
 * the complete base law, and blending never discards a supported tail. */
export function mixEventSizeSigns(base: readonly number[], predicted: readonly number[], blend: number): number[] {
  if (base.length !== 5 || predicted.length !== 4 || !(blend >= 0 && blend <= 1)
    || base.some(v => !Number.isFinite(v) || v < 0) || predicted.some(v => !Number.isFinite(v) || v <= 0)
    || Math.abs(predicted.reduce((s, p) => s + p, 0) - 1) > 1e-8) throw new Error("Invalid size/sign probabilities");
  if (base.slice(0, 4).some(v => !v)) return [...base];
  const active = base.slice(0, 4).reduce((s, p) => s + p, 0);
  return [...predicted.map((p, i) => (1 - blend) * base[i] + blend * active * p), base[4]];
}

export function reweightEventSizeSigns(kernel: readonly MoveAtom[], thresholdLogBps: number, predicted: readonly number[], blend: number): MoveAtom[] {
  const base = eventSizeSignMass(kernel, thresholdLogBps), mixed = mixEventSizeSigns(base, predicted, blend);
  return kernel.map(a => {
    const group = eventSizeSignGroup(a.return, thresholdLogBps);
    return { ...a, probability: a.probability && mixed[group] !== base[group]
      ? Math.max(Number.MIN_VALUE, (a.probability / base[group]) * mixed[group]) : a.probability };
  });
}

/** Project the head's TRAINING-feature probabilities into the existing finite
 * market states. Bellman continuation can then use the new law at every depth.
 * This is still a coarse-state projection: it does not simulate fine features
 * or completed-event memory inside a hypothetical future branch. */
export function projectEventSizeSigns(base: EventDistribution, thresholdLogBps: number,
  rows: readonly { features: number[]; probabilities: number[] }[], blend: number): EventDistribution {
  if (!rows.length || base.runSymmetry || base.hidden || base.meanCalibration) throw new Error("Incompatible event continuation projection");
  const sums = base.kernels.map(() => [0, 0, 0, 0]), counts = base.kernels.map(() => 0);
  for (const row of rows) {
    // Validate full probabilities independently of whether a leaf has support.
    mixEventSizeSigns([0.25, 0.25, 0.25, 0.25, 0], row.probabilities, blend);
    const leaf = eventLeaf(base, row.features); counts[leaf]++;
    for (let g = 0; g < 4; g++) sums[leaf][g] += row.probabilities[g];
  }
  const model: EventDistribution = { ...base,
    kernels: base.kernels.map((kernel, leaf) => counts[leaf]
      ? reweightEventSizeSigns(kernel, thresholdLogBps, sums[leaf].map(p => p / counts[leaf]), blend)
      : kernel.map(a => ({ ...a }))),
    classProbabilities: [] };
  model.classProbabilities = model.kernels.map(kernel => {
    const classes = new Array<number>(15).fill(0);
    for (const a of kernel) classes[eventMoveLabel(a.return, a.duration, model.clock)] += a.probability;
    return classes;
  });
  validateEventDistribution(model);
  return model;
}

export interface EventSizeGateOptions { window: number; strength: number; }
/** A regularized rolling intercept correction in log odds. Only completed
 * post-fit event labels enter the history, paired with their RAW gate forecast.
 * strength/4 is the quadratic penalty precision (binary information units). */
export class EventSizeGateCalibration {
  private history: Array<{ logit: number; large: number }> = [];
  private lastAvailable: number;
  private lastForecast: number;
  constructor(readonly options: EventSizeGateOptions, readonly after: number) {
    if (!Number.isInteger(options.window) || options.window < 1 || !Number.isFinite(options.strength) || options.strength <= 0
      || !Number.isFinite(after)) throw new Error("Invalid size-gate calibration options");
    this.lastAvailable = this.lastForecast = after;
  }
  forecast(probability: number, at: number) {
    if (!(probability > 0 && probability < 1) || !Number.isFinite(at) || at < this.lastAvailable || at < this.lastForecast)
      throw new Error("Invalid size-gate probability or forecast availability");
    this.lastForecast = at;
    const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
    let low = -8, high = 8;
    for (let i = 0; i < 40; i++) {
      const offset = (low + high) / 2;
      const gradient = this.options.strength * offset / 4
        + this.history.reduce((s, h) => s + sigmoid(h.logit + offset) - h.large, 0);
      if (gradient > 0) high = offset; else low = offset;
    }
    const offset = this.history.length ? (low + high) / 2 : 0;
    return { probability: Math.max(1e-6, Math.min(1 - 1e-6, sigmoid(Math.log(probability / (1 - probability)) + offset))),
      offset, count: this.history.length };
  }
  observe(event: { originTime: number; availableAt: number; rawProbability: number; large: boolean }, now: number) {
    if (!Number.isFinite(event.originTime) || !Number.isFinite(event.availableAt) || !Number.isFinite(now)
      || event.originTime < this.after || event.originTime < this.lastAvailable || event.originTime < this.lastForecast
      || event.availableAt <= event.originTime || event.availableAt > now || !(event.rawProbability > 0 && event.rawProbability < 1)
      || typeof event.large !== "boolean") throw new Error("Size-gate updates require ordered completed post-fit events");
    this.history.push({ logit: Math.log(event.rawProbability / (1 - event.rawProbability)), large: Number(event.large) });
    if (this.history.length > this.options.window) this.history.shift();
    this.lastAvailable = event.availableAt;
  }
}

/** Change only the size gate, preserving sign conditionals within each size. */
export function withEventSizeGate(probabilities: readonly number[], large: number): number[] {
  if (probabilities.length !== 4 || probabilities.some(p => !Number.isFinite(p) || p <= 0) || !(large > 0 && large < 1)
    || Math.abs(probabilities.reduce((s, p) => s + p, 0) - 1) > 1e-8) throw new Error("Invalid size-gate mixture");
  const ordinaryMass = probabilities[0] + probabilities[1], largeMass = probabilities[2] + probabilities[3];
  return probabilities.map((p, i) => p * (i < 2 ? (1 - large) / ordinaryMass : large / largeMass));
}
