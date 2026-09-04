import type { MoveAtom } from "./event-distribution.js";
import { eventSizeSignGroup } from "./event-size-sign.js";

export interface EventSeverityHead {
  thresholdLogBps: number;
  meanLogRv5: number; scaleLogRv5: number;
  /** Intercept, standardized log volatility, candidate-sign contrast. */
  coefficients: number[];
  penalty: number; samples: number; iterations: number; loss: number;
}
export interface EventSeveritySample { rv5Bps: number; return: number; }

/** Positive excess-magnitude mean T * (1 + exp(eta)). Both candidate signs
 * are evaluated from known volatility; the actual future sign is never input. */
export function predictEventSeverity(head: EventSeverityHead, rv5Bps: number): number[] {
  if (!Number.isFinite(rv5Bps) || rv5Bps < 0) throw new Error("Invalid severity volatility");
  const z = Math.max(-5, Math.min(5, (Math.log1p(rv5Bps) - head.meanLogRv5) / head.scaleLogRv5));
  return [-1, 1].map(sign => head.thresholdLogBps * (1 + Math.exp(Math.max(-30, Math.min(30,
    head.coefficients[0] + head.coefficients[1] * z + head.coefficients[2] * sign)))));
}

/** Three-parameter log-link mean regression, on completed large moves only.
 * The mean loss mu - y*log(mu) targets E[y|x] for nonnegative continuous y;
 * it does not assert a Poisson count distribution. Volatility normalization
 * and the threshold are fitted before evaluation. Slopes have an L2 penalty. */
export function trainEventSeverity(samples: readonly EventSeveritySample[], thresholdLogBps: number, penalty: number): EventSeverityHead {
  if (!(thresholdLogBps > 0) || !Number.isFinite(thresholdLogBps) || !(penalty > 0) || !Number.isFinite(penalty)
    || samples.some(s => !Number.isFinite(s.rv5Bps) || s.rv5Bps < 0 || !Number.isFinite(s.return) || s.return <= -1)) throw new Error("Invalid severity training data");
  const rows = samples.filter(s => eventSizeSignGroup(s.return, thresholdLogBps) === 2 || eventSizeSignGroup(s.return, thresholdLogBps) === 3);
  if (rows.length < 3) throw new Error("Insufficient large-move severity targets");
  const logs = rows.map(r => Math.log1p(r.rv5Bps));
  const meanLogRv5 = logs.reduce((s, v) => s + v, 0) / rows.length;
  const scaleLogRv5 = Math.max(1e-8, Math.sqrt(logs.reduce((s, v) => s + (v - meanLogRv5) ** 2, 0) / rows.length));
  const x = rows.map((r, i) => [1, Math.max(-5, Math.min(5, (logs[i] - meanLogRv5) / scaleLogRv5)), Math.sign(r.return)]);
  const y = rows.map(r => Math.max(0, Math.abs(Math.log1p(r.return)) * 1e4 / thresholdLogBps - 1));
  let beta = [Math.log(Math.max(1e-8, y.reduce((s, v) => s + v, 0) / rows.length)), 0, 0], iterations = 0;
  const objective = (b: number[]) => x.reduce((sum, row, i) => {
    const eta = row.reduce((s, v, j) => s + v * b[j], 0);
    return sum + Math.exp(eta) - y[i] * eta;
  }, 0) / rows.length + penalty * (b[1] ** 2 + b[2] ** 2) / 2;
  let loss = objective(beta);
  for (; iterations < 40; iterations++) {
    const matrix = Array.from({ length: 3 }, () => [0, 0, 0, 0]);
    for (let i = 0; i < rows.length; i++) {
      const mu = Math.exp(x[i].reduce((s, v, j) => s + v * beta[j], 0));
      for (let j = 0; j < 3; j++) {
        matrix[j][3] += x[i][j] * (mu - y[i]) / rows.length;
        for (let k = 0; k < 3; k++) matrix[j][k] += x[i][j] * x[i][k] * mu / rows.length;
      }
    }
    for (let j = 1; j < 3; j++) { matrix[j][j] += penalty; matrix[j][3] += penalty * beta[j]; }
    matrix[0][0] += 1e-12;
    if (Math.max(...matrix.map(r => Math.abs(r[3]))) < 1e-8) break;
    for (let j = 0; j < 3; j++) {
      let pivot = j;
      for (let k = j + 1; k < 3; k++) if (Math.abs(matrix[k][j]) > Math.abs(matrix[pivot][j])) pivot = k;
      [matrix[j], matrix[pivot]] = [matrix[pivot], matrix[j]];
      const divisor = matrix[j][j];
      if (!(Math.abs(divisor) > 1e-14)) throw new Error("Singular severity mean fit");
      for (let k = j; k < 4; k++) matrix[j][k] /= divisor;
      for (let k = 0; k < 3; k++) if (k !== j) {
        const multiplier = matrix[k][j];
        for (let l = j; l < 4; l++) matrix[k][l] -= multiplier * matrix[j][l];
      }
    }
    let accepted = false;
    for (let step = 1; step >= 1 / 1024; step /= 2) {
      const next = beta.map((v, j) => v - step * matrix[j][3]), nextLoss = objective(next);
      if (Number.isFinite(nextLoss) && nextLoss < loss) { beta = next; loss = nextLoss; accepted = true; break; }
    }
    if (!accepted) break;
  }
  return { thresholdLogBps, meanLogRv5, scaleLogRv5, coefficients: beta, penalty, samples: rows.length, iterations, loss };
}

/** Conditional expected absolute log returns in the two large/sign groups. */
export function eventSeverityMeans(kernel: readonly MoveAtom[], thresholdLogBps: number): Array<number | null> {
  return [2, 3].map(group => {
    let mass = 0, sum = 0;
    for (const a of kernel) if (eventSizeSignGroup(a.return, thresholdLogBps) === group) {
      mass += a.probability; sum += a.probability * Math.abs(Math.log1p(a.return)) * 1e4;
    }
    return mass ? sum / mass : null;
  });
}

/** Exponential tilt WITHIN each large/sign group. Its total probability and
 * every joint return/extremum/duration/successor value are retained. Impossible
 * target means are clamped inside existing support; no new path is invented.
 * Positive underflowed tails retain minimal mass so ruin remains represented. */
export function tiltEventSeverity(kernel: readonly MoveAtom[], thresholdLogBps: number, targets: readonly number[], blend: number): MoveAtom[] {
  if (targets.length !== 2 || targets.some(v => !Number.isFinite(v) || v < thresholdLogBps)
    || !(blend >= 0 && blend <= 1) || !(thresholdLogBps > 0) || !Number.isFinite(thresholdLogBps)) throw new Error("Invalid conditional severity targets");
  const result = kernel.map(a => ({ ...a }));
  if (blend === 0) return result;
  for (const group of [2, 3]) {
    const indices = kernel.flatMap((a, i) => a.probability > 0 && eventSizeSignGroup(a.return, thresholdLogBps) === group ? [i] : []);
    if (indices.length < 2) continue;
    const values = indices.map(i => Math.abs(Math.log1p(kernel[i].return)) * 1e4);
    const mass = indices.reduce((s, i) => s + kernel[i].probability, 0);
    const minimum = Math.min(...values), maximum = Math.max(...values), range = maximum - minimum;
    if (range < 1e-8) continue;
    const current = indices.reduce((s, i, j) => s + kernel[i].probability * values[j], 0) / mass;
    const target = Math.max(minimum + range * 1e-10, Math.min(maximum - range * 1e-10, (1 - blend) * current + blend * targets[group - 2]));
    if (Math.abs(target - current) < 1e-10) continue;
    const normalized = values.map(v => (v - minimum) / range), logs = indices.map(i => Math.log(kernel[i].probability));
    const weights = (theta: number) => {
      const logWeights = logs.map((v, j) => v + theta * normalized[j]), top = Math.max(...logWeights);
      const raw = logWeights.map(v => Math.exp(v - top)), sum = raw.reduce((s, v) => s + v, 0);
      return raw.map(v => v / sum);
    };
    let low = -2048, high = 2048;
    for (let iteration = 0; iteration < 55; iteration++) {
      const middle = (low + high) / 2, p = weights(middle);
      const mean = p.reduce((s, v, j) => s + v * values[j], 0);
      if (mean < target) low = middle; else high = middle;
    }
    const p = weights((low + high) / 2);
    for (let j = 0; j < indices.length; j++) result[indices[j]].probability = Math.max(Number.MIN_VALUE, mass * p[j]);
  }
  return result;
}
