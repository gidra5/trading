import type { EventSignHead } from "./event-sign.js";
import { predictEventSizeSigns, type EventSizeSignHead } from "./event-size-sign.js";

/** Each row fixes the planner, its conditional joint outcome laws, and a set
 * of exposure/depth probes. Only the four forecast probabilities are learned.
 * coefficients[action][group] and targets[action] are log-wealth advantages
 * over cash, so action-independent continuation values cannot dominate loss. */
export interface EventValueHeadRow {
  features: number[];
  coefficients: number[][];
  offsets: number[];
  targets: number[];
}
export interface EventValueSizeSignHead extends EventSizeSignHead {
  valueFit: { penalty: number; iterations: number; samples: number; scale: number; loss: number; initialLoss: number };
}

export function eventValueHeadMse(head: EventSizeSignHead, rows: readonly EventValueHeadRow[]): number {
  let squared = 0, count = 0;
  for (const row of rows) {
    const p = predictEventSizeSigns(head, row.features);
    for (let j = 0; j < row.targets.length; j++) {
      const prediction = row.offsets[j] + row.coefficients[j].reduce((s, v, g) => s + v * p[g], 0);
      squared += (prediction - row.targets[j]) ** 2; count++;
    }
  }
  if (!count) throw new Error("No value-head evaluation targets");
  return squared / count;
}

/** One value-aware model improvement with frozen continuation values, inspired
 * by IterVAML. This is not a converged alternating model/planner iteration.
 * Delta parameters are penalized around the incumbent sign/size head, including
 * intercepts; standardization and all within-group joint paths stay fixed. */
export function trainEventValueHead(base: EventSizeSignHead, rows: readonly EventValueHeadRow[], penalty: number): EventValueSizeSignHead {
  const components = [base.gate, base.ordinarySign, base.largeSign];
  const width = Math.max(...components.map(h => h.means.length)), probes = rows[0]?.targets.length;
  if (!probes || rows.length < 2 || !(penalty > 0) || !Number.isFinite(penalty)
    || rows.some(r => r.features.length !== width || r.targets.length !== probes || r.offsets.length !== probes
      || r.coefficients.length !== probes || r.coefficients.some(c => c.length !== 4)
      || [...r.features, ...r.targets, ...r.offsets, ...r.coefficients.flat()].some(v => !Number.isFinite(v)))) throw new Error("Invalid value-head data");
  const scale = Math.max(1e-6, Math.sqrt(rows.reduce((s, r) => s + r.targets.reduce((s, y) => s + y * y, 0), 0) / (rows.length * probes)));
  const starts = [0, base.gate.coefficients.length + 1, base.gate.coefficients.length + base.ordinarySign.coefficients.length + 2];
  const original = components.flatMap(h => [h.intercept, ...h.coefficients]);
  const inputs = rows.map(row => components.map(h => [1, ...row.features.slice(0, h.means.length)
    .map((v, i) => Math.max(-5, Math.min(5, (v - h.means[i]) / h.scales[i])))]));
  const coefficients = rows.map(r => r.coefficients.map(c => c.map(v => v / scale)));
  const targets = rows.map(r => r.targets.map((y, j) => (y - r.offsets[j]) / scale));
  const calculate = (delta: number[], gradient?: number[]) => {
    let loss = penalty * delta.reduce((s, v) => s + v * v, 0) / 2;
    if (gradient) for (let k = 0; k < delta.length; k++) gradient[k] = penalty * delta[k];
    for (let i = 0; i < rows.length; i++) {
      const z = components.map((_, h) => inputs[i][h].reduce((s, x, k) => s + x * (original[starts[h] + k] + delta[starts[h] + k]), 0));
      const q = z.map(v => Math.max(1e-6, Math.min(1 - 1e-6, 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, v)))))));
      const [g, o, l] = q, p = [(1 - g) * (1 - o), (1 - g) * o, g * (1 - l), g * l];
      const derivatives = [[-(1 - o), -o, 1 - l, l], [-(1 - g), 1 - g, 0, 0], [0, 0, -g, g]];
      const logitGradient = [0, 0, 0];
      for (let j = 0; j < probes; j++) {
        const c = coefficients[i][j], error = c.reduce((s, v, k) => s + v * p[k], 0) - targets[i][j];
        loss += error * error / (2 * rows.length * probes);
        if (gradient) for (let h = 0; h < 3; h++) logitGradient[h] += error * c.reduce((s, v, k) => s + v * derivatives[h][k], 0) / (rows.length * probes);
      }
      if (gradient) for (let h = 0; h < 3; h++) {
        const slope = q[h] > 1e-6 && q[h] < 1 - 1e-6 ? q[h] * (1 - q[h]) : 0;
        for (let k = 0; k < inputs[i][h].length; k++) gradient[starts[h] + k] += logitGradient[h] * slope * inputs[i][h][k];
      }
    }
    return loss;
  };
  let delta = original.map(() => 0), iterations = 0, loss = calculate(delta), step = 1;
  const initialLoss = loss;
  for (; iterations < 400; iterations++) {
    const gradient = delta.map(() => 0); calculate(delta, gradient);
    if (Math.max(...gradient.map(Math.abs)) < 1e-7) break;
    let accepted = false;
    for (let tries = 0; tries < 20; tries++) {
      const next = delta.map((v, k) => v - step * gradient[k]), nextLoss = calculate(next);
      if (nextLoss < loss) { delta = next; loss = nextLoss; step = Math.min(8, step * 1.25); accepted = true; break; }
      step /= 2;
    }
    if (!accepted) break;
  }
  const updated = components.map((h, j): EventSignHead => ({ ...h,
    objective: "joint-value-error", penalty, samples: rows.length, iterations, loss,
    intercept: h.intercept + delta[starts[j]], coefficients: h.coefficients.map((v, k) => v + delta[starts[j] + k + 1]) }));
  return { ...base, gate: updated[0], ordinarySign: updated[1], largeSign: updated[2],
    valueFit: { penalty, iterations, samples: rows.length, scale, loss, initialLoss } };
}
