import type { MoveAtom, MoveSample } from "./event-distribution.js";

/** A separate active-sign head. Ordinary fitting ignores target magnitudes.
 * A weighted head estimates the reweighted class probability, not P(up). */
export interface EventSignHead {
  means: number[]; scales: number[]; coefficients: number[]; intercept: number;
  penalty: number; samples: number; iterations: number; loss: number;
  objective?: "joint-value-error" | "weighted-sign";
}
const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, z))));
const bounded = (p: number) => Math.max(1e-6, Math.min(1 - 1e-6, p));
const coordinates = (head: Pick<EventSignHead, "means" | "scales">, x: readonly number[]) =>
  x.map((v, i) => Math.max(-5, Math.min(5, (v - head.means[i]) / head.scales[i])));

export function predictEventSign(head: EventSignHead, features: readonly number[]): number {
  if (features.length !== head.means.length || features.some(v => !Number.isFinite(v))) throw new Error("Invalid sign features");
  return bounded(sigmoid(head.intercept + coordinates(head, features).reduce((s, x, i) => s + x * head.coefficients[i], 0)));
}

/** Penalized logistic regression, fitted only on nonzero completed targets.
 * The objective is average cross entropy + penalty * squared slopes / 2.
 * Train-only standardization and bounded coordinates match the small ridge
 * baseline; damped Newton steps keep fitting deterministic and inexpensive. */
export function trainEventSign(samples: readonly Pick<MoveSample, "features" | "return">[], penalty: number,
  weights?: readonly number[]): EventSignHead {
  const width = samples[0]?.features.length ?? 0;
  if (!width || !(penalty > 0) || !Number.isFinite(penalty) || samples.some(s => s.features.length !== width
    || [...s.features, s.return].some(v => !Number.isFinite(v)))) throw new Error("Invalid sign training data");
  if (weights && (weights.length !== samples.length || weights.some(w => !Number.isFinite(w) || w < 0))) throw new Error("Invalid sign sample weights");
  const active = samples.map((s, i) => ({ sample: s, weight: weights?.[i] ?? 1 })).filter(r => r.sample.return !== 0 && r.weight > 0);
  const rows = active.map(r => r.sample), totalWeight = active.reduce((s, r) => s + r.weight, 0);
  if (!Number.isFinite(totalWeight) || !(totalWeight > 0)) throw new Error("Invalid active sign weight total");
  // Normalize to mean one so the penalty has the same meaning under any
  // change in weight units. Ordinary fits retain their exact arithmetic.
  const normalized = weights ? active.map(r => r.weight / totalWeight * rows.length) : undefined;
  if (rows.length < 2) throw new Error("Insufficient active sign targets");
  const means = Array.from({ length: width }, (_, f) => rows.reduce((s, r) => s + r.features[f], 0) / rows.length);
  const scales = means.map((m, f) => Math.max(1e-8, Math.sqrt(rows.reduce((s, r) => s + (r.features[f] - m) ** 2, 0) / rows.length)));
  const x = rows.map(r => [1, ...coordinates({ means, scales }, r.features)]), y = rows.map(r => Number(r.return > 0));
  const positive = y.reduce((s, v, i) => s + v * (normalized?.[i] ?? 1), 0), p = (positive + 0.5) / (rows.length + 1);
  const bothClasses = y.some(v => v === 0) && y.some(v => v === 1);
  let beta = [Math.log(p / (1 - p)), ...new Array<number>(width).fill(0)], iterations = 0;
  const objective = (b: number[]) => x.reduce((sum, row, i) => {
    const z = row.reduce((s, v, j) => s + v * b[j], 0);
    return normalized ? sum + normalized[i] * (Math.max(z, 0) - y[i] * z + Math.log1p(Math.exp(-Math.abs(z))))
      : sum + Math.max(z, 0) - y[i] * z + Math.log1p(Math.exp(-Math.abs(z)));
  }, 0) / rows.length + penalty * b.slice(1).reduce((s, v) => s + v * v, 0) / 2;
  let loss = objective(beta);
  // A constant target has no finite unpenalized-intercept MLE. Keep its
  // half-count estimate rather than eliminating the unobserved tail sign.
  if (bothClasses) for (; iterations < 40; iterations++) {
    const n = width + 1, h = Array.from({ length: n }, () => new Array<number>(n + 1).fill(0));
    for (let i = 0; i < rows.length; i++) {
      const row = x[i], q = sigmoid(row.reduce((s, v, j) => s + v * beta[j], 0));
      for (let j = 0; j < n; j++) {
        const weight = normalized?.[i] ?? 1;
        h[j][n] += row[j] * (q - y[i]) * weight / rows.length;
        for (let k = 0; k <= j; k++) h[j][k] += row[j] * row[k] * q * (1 - q) * weight / rows.length;
      }
    }
    for (let j = 0; j < n; j++) {
      for (let k = j + 1; k < n; k++) h[j][k] = h[k][j];
      h[j][j] += j ? penalty : 1e-10;
      if (j) h[j][n] += penalty * beta[j];
    }
    if (Math.max(...h.map(r => Math.abs(r[n]))) < 1e-8) break;
    for (let j = 0; j < n; j++) {
      let pivot = j;
      for (let k = j + 1; k < n; k++) if (Math.abs(h[k][j]) > Math.abs(h[pivot][j])) pivot = k;
      [h[j], h[pivot]] = [h[pivot], h[j]];
      const divisor = h[j][j];
      if (!(Math.abs(divisor) > 1e-14)) throw new Error("Singular sign fit");
      for (let k = j; k <= n; k++) h[j][k] /= divisor;
      for (let k = 0; k < n; k++) if (k !== j) {
        const factor = h[k][j];
        for (let l = j; l <= n; l++) h[k][l] -= factor * h[j][l];
      }
    }
    let step = 1, accepted = false;
    while (step >= 1 / 1024) {
      const next = beta.map((b, j) => b - step * h[j][n]), nextLoss = objective(next);
      if (nextLoss < loss) { beta = next; loss = nextLoss; accepted = true; break; }
      step /= 2;
    }
    if (!accepted) break;
  }
  return { means, scales, intercept: beta[0], coefficients: beta.slice(1), penalty, samples: rows.length, iterations, loss,
    ...(weights ? { objective: "weighted-sign" as const } : {}) };
}

/** If q = E[|r| 1(r>0)|x] / E[|r||x], recover a sign mixture using
 * the existing conditional mean magnitudes. The resulting law matches the
 * predicted gain/loss balance, conditional on those magnitude estimates.
 * q itself is NOT an ordinary up probability. */
export function eventProbabilityFromReturnWeight(q: number, positiveMean: number, negativeMean: number): number {
  if (!(q > 0 && q < 1) || !Number.isFinite(positiveMean) || !Number.isFinite(negativeMean)
    || !(positiveMean > 0) || !(negativeMean < 0)) throw new Error("Invalid return-weighted sign inversion");
  const negativeMagnitude = -negativeMean, scale = Math.max(positiveMean, negativeMagnitude);
  const numerator = q * (negativeMagnitude / scale);
  return bounded(numerator / (numerator + (1 - q) * (positiveMean / scale)));
}

export function eventSignMass(kernel: readonly MoveAtom[]): { negative: number; zero: number; positive: number; probability: number } {
  let negative = 0, zero = 0, positive = 0;
  for (const a of kernel) {
    if (a.return < 0) negative += a.probability;
    else if (a.return > 0) positive += a.probability;
    else zero += a.probability;
  }
  return { negative, zero, positive, probability: positive + negative ? positive / (positive + negative) : 0.5 };
}

/** P(sign | x) times the old JOINT conditional law given sign and state.
 * Zero mass and all return/extrema/duration/successor values are unchanged.
 * A missing sign has no conditional law to reweight, so retain the base law. */
export function reweightEventSigns(kernel: readonly MoveAtom[], probability: number): MoveAtom[] {
  if (!Number.isFinite(probability) || !(probability > 0 && probability < 1)) throw new Error("Sign probability must be strictly between zero and one");
  const { negative, positive } = eventSignMass(kernel), active = negative + positive;
  if (!negative || !positive) return kernel.map(a => ({ ...a }));
  return kernel.map(a => ({ ...a, probability: a.return === 0 || !a.probability ? a.probability
    : Math.max(Number.MIN_VALUE, a.probability * (a.return > 0 ? active * probability / positive : active * (1 - probability) / negative)) }));
}
