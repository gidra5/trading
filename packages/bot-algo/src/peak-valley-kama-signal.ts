import { signalBeyondFriction } from "./signal-memory.js";

export type PeakValleyKamaClampMode = "deadband" | "hysteresis" | "hold";

export interface PeakValleyKamaSignalMemory {
  candidate: number;
  accepted: number;
  lastSignalPrice: number | null;
}

export interface PeakValleyKamaSignalStep {
  candidate: number;
  accepted: number;
  lastSignalPrice: number | null;
  sourceEdge: boolean;
  transitionRequested: boolean;
  beyondFriction: boolean;
  transitionAccepted: boolean;
}

/**
 * Production peak/valley threshold state machine. Values remain continuous so
 * diagnostics retain the observed rate; only their signs define exposure state.
 */
export function clampPeakValleyKamaRate(
  value: number,
  previous: number,
  mode: PeakValleyKamaClampMode,
  innerThresholdRatio: number,
  low: number,
  high: number,
): number {
  if (mode === "hold") return value > high || value < -low ? value : previous;
  const release = low * clamp01(innerThresholdRatio);
  if (
    mode === "hysteresis"
    && previous !== 0
    && Math.sign(previous) === Math.sign(value)
    && Math.abs(value) >= release
  ) return value;
  const threshold = mode === "hysteresis" && previous === 0 ? high : low;
  return Math.abs(value) > threshold ? value : 0;
}

/**
 * Shared causal signal memory used by live trading and batch evaluation.
 * A rejected source edge is consumed: it is not retried until the source
 * changes state again.
 */
export function advancePeakValleyKamaSignal(
  rate: number,
  price: number,
  memory: PeakValleyKamaSignalMemory,
  options: {
    mode: PeakValleyKamaClampMode;
    innerThresholdRatio: number;
    thresholdLow: number;
    thresholdHigh: number;
    signalFriction: number;
    transitionAllowed?: boolean;
  },
): PeakValleyKamaSignalStep {
  const candidate = clampPeakValleyKamaRate(
    rate,
    memory.candidate,
    options.mode,
    options.innerThresholdRatio,
    options.thresholdLow,
    options.thresholdHigh,
  );
  return resolvePeakValleyKamaSignal(candidate, price, memory, {
    sourceEdge: Math.sign(candidate) !== Math.sign(memory.candidate),
    signalFriction: options.signalFriction,
    transitionAllowed: options.transitionAllowed,
  });
}

/** Resolve signal memory after any optional causal filters transform a source state. */
export function resolvePeakValleyKamaSignal(
  candidate: number,
  price: number,
  memory: PeakValleyKamaSignalMemory,
  options: {
    sourceEdge: boolean;
    signalFriction: number;
    transitionAllowed?: boolean;
  },
): PeakValleyKamaSignalStep {
  const transitionRequested = options.sourceEdge
    && Math.sign(candidate) !== Math.sign(memory.accepted);
  const beyondFriction = !transitionRequested || signalBeyondFriction(
    price,
    memory.lastSignalPrice,
    Math.max(0, options.signalFriction),
  );
  const transitionAccepted = transitionRequested
    && options.transitionAllowed !== false
    && beyondFriction;
  const accepted = Math.sign(candidate) === Math.sign(memory.accepted) || transitionAccepted
    ? candidate
    : memory.accepted;
  return {
    candidate,
    accepted,
    lastSignalPrice: transitionAccepted ? price : memory.lastSignalPrice,
    sourceEdge: options.sourceEdge,
    transitionRequested,
    beyondFriction,
    transitionAccepted,
  };
}

export function peakValleyKamaRate(
  delta: number,
  value: number,
  seconds: number,
  relative: boolean,
  mode: "relative" | "log",
): number {
  if (!relative) return delta / seconds;
  if (value <= 0) return 0;
  if (mode !== "log") return delta / value / seconds;
  const previous = value - delta;
  return previous > 0 ? Math.log(value / previous) / seconds : 0;
}

export function peakValleyKamaThresholdAdjustment(
  noise: number,
  options: {
    response: "proportional" | "inverse";
    multiplier: number;
    inverseMax: number;
    inverseNoiseScale: number;
  },
): number {
  if (options.response !== "inverse") {
    return Math.max(0, noise) * Math.max(0, options.multiplier);
  }
  return Math.max(0, options.inverseMax)
    / (1 + Math.max(0, noise) / Math.max(Number.EPSILON, options.inverseNoiseScale));
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}
