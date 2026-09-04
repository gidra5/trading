import { chooseEventTrade, eventHolding, type EventAccount, type EventCosts, type EventPolicy } from "./event-log-policy.js";
import type { MoveAtom } from "./event-distribution.js";
import { predictEventValueBoost, trainEventValueBoost, type EventValueBoost, type EventValueBoostOptions } from "./event-value-boost.js";

export interface EventValueTransition {
  features: number[]; nextFeatures: number[]; leaf: number; nextLeaf: number;
  move: Pick<MoveAtom, "return" | "low" | "high" | "duration">;
}
export interface FittedEventValue {
  contract: "event-fitted-value-v1";
  costs: EventCosts; targets: number[]; exposures: number[]; equities: number[]; prices: number[];
  means: number[]; scales: number[]; penalty: number; samples: number;
  /** Non-cryptographic consistency token for resuming these exact transitions.
   * Older research artifacts without a token remain inference-only. */
  trainingSignature?: string;
  /** Observed-path targets use either earlier fitted actions or a fixed hold/cap controller. */
  targetMode?: "sampled-path" | "minimum-turnover";
  pathHorizon?: number;
  /** Fixed external continuation heads, for held-out policy-evaluation targets. */
  continuationSignature?: string;
  /** Finite roots at which at least one positive-probability path liquidates.
   * null denotes an unbounded side. These guards remain independent of regression. */
  limits: Array<{ long: number | null; short: number | null }>;
  /** Each grid cell has a linear head and optional shared-partition residual
   * correction. Null forbids the cell regardless of the residual prediction. */
  tables: Array<{ depth: number; coefficients: Array<number[] | null>; correction?: EventValueBoost; trainingMse: number }>;
}

function coordinates(p: Pick<FittedEventValue, "means" | "scales">, features: readonly number[]): number[] {
  if (features.length !== p.means.length || features.some(v => !Number.isFinite(v))) throw new Error("Invalid fitted-value inputs");
  return [1, ...features.map((v, i) => Math.max(-5, Math.min(5, (v - p.means[i]) / p.scales[i])))];
}
function bracket(axis: readonly number[], value: number): [number, number, number] {
  if (value <= axis[0]) return [0, 0, 0];
  const last = axis.length - 1; if (value >= axis[last]) return [last, last, 0];
  let low = 0, high = last;
  while (high - low > 1) { const m = (low + high) >> 1; if (axis[m] <= value) low = m; else high = m; }
  return [low, high, (value - axis[low]) / (axis[high] - axis[low])];
}
const offset = (p: FittedEventValue, w: number, price: number, x: number) => (w * p.prices.length + price) * p.exposures.length + x;
function lookup(p: FittedEventValue, values: readonly number[], account: EventAccount, leaf: number): number {
  const limit = p.limits[leaf]; if (!limit) throw new Error("Invalid fitted-value guard state");
  if ((account.exposure > 0 && limit.long !== null && account.exposure >= limit.long)
    || (account.exposure < 0 && limit.short !== null && -account.exposure >= limit.short)) return -Infinity;
  const axes = [bracket(p.equities, account.equity), bracket(p.prices, account.price), bracket(p.exposures, account.exposure)];
  let value = 0;
  for (let w = 0; w < 2; w++) for (let price = 0; price < 2; price++) for (let x = 0; x < 2; x++) {
    const weight = (w ? axes[0][2] : 1 - axes[0][2]) * (price ? axes[1][2] : 1 - axes[1][2]) * (x ? axes[2][2] : 1 - axes[2][2]);
    if (weight) value += weight * values[offset(p, axes[0][w], axes[1][price], axes[2][x])];
  }
  return value;
}
export function predictEventHoldingGrid(p: FittedEventValue, features: readonly number[], depth: number): number[] {
  const table = p.tables[depth - 1]; if (!table) throw new Error("Unavailable fitted-value depth");
  const x = coordinates(p, features);
  const correction = table.correction ? predictEventValueBoost(table.correction, x) : undefined;
  return table.coefficients.map((beta, i) => {
    if (!beta) return -Infinity;
    const value = beta.reduce((s, b, j) => s + b * x[j], 0) + (correction?.[i % correction.length] ?? 0);
    // Cash can remain cash through every future event. Linear extrapolation
    // must not undervalue that feasible zero-return continuation.
    return p.exposures[i % p.exposures.length] === 0 ? Math.max(0, value) : value;
  });
}
export function fittedEventHolding(p: FittedEventValue, grid: readonly number[], account: EventAccount, leaf: number): number {
  if (grid.length !== p.equities.length * p.prices.length * p.exposures.length) throw new Error("Invalid fitted holding grid");
  return lookup(p, grid, account, leaf);
}
export function decideFittedEvent(p: FittedEventValue, features: readonly number[], account: EventAccount, depth: number, leaf: number) {
  const values = predictEventHoldingGrid(p, features, depth);
  return chooseEventTrade(p, account, held => lookup(p, values, held, leaf));
}

/** Compare complete interpolated controller values on each feasible action.
 * Maximizing grid cells before interpolation would splice different policies
 * into an artificial account value. Exact ties retain the first controller. */
export function decideFittedEventControllers(policies: readonly FittedEventValue[], features: readonly number[], account: EventAccount, depth: number, leaf: number) {
  if (!policies.length) throw new Error("Missing fitted controllers");
  const first = policies[0];
  const contract = (p: FittedEventValue) => JSON.stringify([p.contract, p.costs, p.targets, p.equities, p.prices, p.exposures, p.limits]);
  if (policies.some(p => contract(p) !== contract(first))) throw new Error("Incompatible fitted controller accounts");
  const grids = policies.map(p => predictEventHoldingGrid(p, features, depth));
  const values = (a: EventAccount) => policies.map((p, i) => lookup(p, grids[i], a, leaf));
  const decision = chooseEventTrade(first, account, a => Math.max(...values(a))), holdingValues = values(decision);
  const controller = holdingValues.reduce((best, value, i) => value > holdingValues[best] + 1e-12 ? i : best, 0);
  return { decision, controller, holdingValues };
}

function pathLimits(base: EventPolicy): FittedEventValue["limits"] {
  const c = base.costs, m = c.maintenanceMargin;
  return base.model.kernels.map(kernel => {
    let long = Infinity, short = Infinity;
    for (const a of kernel) if (a.probability > 0) {
      const b = c.longBorrowBpsPerDay / 1e4 * a.duration / 1440;
      const slope = (1 - m) * a.low - m, unborrowed = slope < 0 ? -1 / slope : Infinity;
      const longRoot = unborrowed <= 1 ? unborrowed : b - slope > 0 ? (1 + b) / (b - slope) : Infinity;
      const shortSlope = a.high + c.shortBorrowBpsPerDay / 1e4 * a.duration / 1440 + m * (1 + a.high);
      long = Math.min(long, longRoot); if (shortSlope > 0) short = Math.min(short, 1 / shortSlope);
    }
    return { long: Number.isFinite(long) ? long : null, short: Number.isFinite(short) ? short : null };
  });
}

function transitionSignature(rows: readonly EventValueTransition[]): string {
  // Two independent 32-bit byte hashes over every IEEE-754 training value.
  // This catches accidental data changes; it is not an authenticity mechanism.
  const bytes = new DataView(new ArrayBuffer(8)); let first = 0x811c9dc5, second = 0x9e3779b9;
  for (const row of rows) for (const value of [...row.features, ...row.nextFeatures, row.leaf, row.nextLeaf,
    row.move.return, row.move.low, row.move.high, row.move.duration]) {
    bytes.setFloat64(0, value, true);
    for (let i = 0; i < 8; i++) { const byte = bytes.getUint8(i); first = Math.imul(first ^ byte, 16777619); second = Math.imul(second ^ byte, 2246822519); }
  }
  return `${rows.length}:${rows[0].features.length}:${(first >>> 0).toString(16)}:${(second >>> 0).toString(16)}`;
}

/** Fitted holding-value iteration on complete observed market transitions.
 * H_d(s,x) <- regression of log G(x,event) + V_(d-1)(observed s', account').
 * The maximum is over the PREVIOUS fitted conditional value, never over a
 * realized future path. All equity/price/exposure axes retain the base grid.
 * This is regularized finite-horizon FQI, not a convergence certificate. */
export function trainEventFittedValue(base: EventPolicy, rows: readonly EventValueTransition[], penalty: number, depths: number,
  boost?: EventValueBoostOptions, resume?: FittedEventValue,
  rollout?: { following: readonly (readonly EventValueTransition[])[]; policies?: readonly FittedEventValue[];
    /** Hold inventory between events, making only the cheapest feasible cap reduction. */
    minimumTurnover?: true }): FittedEventValue {
  const width = rows[0]?.features.length ?? 0;
  const pathHorizon = rollout ? (rollout.following[0]?.length ?? 0) + 1 : undefined;
  if (rollout?.minimumTurnover && rollout.policies) throw new Error("Minimum-turnover continuation cannot use fitted continuation heads");
  if (rollout && (rollout.following.length !== rows.length || pathHorizon! < depths
    || rollout.following.some((tail, i) => tail.length + 1 !== pathHorizon || tail.some((r, j) => {
      const previous = j ? tail[j - 1] : rows[i];
      return r.leaf !== previous.nextLeaf || JSON.stringify(r.features) !== JSON.stringify(previous.nextFeatures);
    })))) throw new Error("Invalid sampled continuation paths");
  const completeRows = rollout ? rows.flatMap((r, i) => [r, ...rollout.following[i]]) : rows;
  if (rows.length < 2 || !width || !(penalty > 0) || !Number.isFinite(penalty) || !Number.isInteger(depths) || depths < 1
    || completeRows.some(r => r.features.length !== width || r.nextFeatures.length !== width
      || [...r.features, ...r.nextFeatures, ...Object.values(r.move)].some(v => !Number.isFinite(v))
      || !(r.move.return > -1) || !(r.move.low > -1) || r.move.low > Math.min(0, r.move.return) || r.move.high < Math.max(0, r.move.return)
      || r.move.duration <= 0 || !base.model.kernels[r.leaf] || !base.model.kernels[r.nextLeaf])) throw new Error("Invalid fitted-value transitions/settings");
  const means = Array.from({ length: width }, (_, f) => rows.reduce((s, r) => s + r.features[f], 0) / rows.length);
  const scales = means.map((mean, f) => Math.max(1e-8, Math.sqrt(rows.reduce((s, r) => s + (r.features[f] - mean) ** 2, 0) / rows.length)));
  const p: FittedEventValue = { contract: "event-fitted-value-v1", costs: { ...base.costs }, targets: [...base.targets],
    equities: [...base.equities], prices: [...base.prices], exposures: [...base.exposures], means, scales, penalty, samples: rows.length,
    limits: pathLimits(base), trainingSignature: transitionSignature(completeRows),
    ...(rollout ? { targetMode: rollout.minimumTurnover ? "minimum-turnover" as const : "sampled-path" as const, pathHorizon } : {}), tables: [] };
  if (rollout?.policies) {
    const contract = (v: FittedEventValue) => JSON.stringify([v.contract, v.costs, v.targets, v.equities, v.prices, v.exposures, v.limits]);
    const hashes = new Map<FittedEventValue, string>();
    const hash = (s: string) => {
      let a = 0x811c9dc5, b = 0x9e3779b9;
      for (let i = 0; i < s.length; i++) { a = Math.imul(a ^ s.charCodeAt(i), 16777619); b = Math.imul(b ^ s.charCodeAt(i), 2246822519); }
      return `${(a >>> 0).toString(16)}:${(b >>> 0).toString(16)}`;
    };
    if (rollout.policies.length !== rows.length) throw new Error("Invalid external continuation policy count");
    for (const v of rollout.policies) if (!hashes.has(v)) {
      if (contract(v) !== contract(p) || v.means.length !== width || v.scales.length !== width
        || v.means.some(x => !Number.isFinite(x)) || v.scales.some(x => !Number.isFinite(x) || x <= 0)
        || v.tables.length < Math.max(1, depths - 1) || v.tables.some((t, i) => t.depth !== i + 1
          || t.coefficients.length !== p.equities.length * p.prices.length * p.exposures.length
          || t.coefficients.some(b => b !== null && (b.length !== width + 1 || b.some(x => !Number.isFinite(x))))))
        throw new Error("Incompatible external continuation policy");
      hashes.set(v, hash(JSON.stringify(v)));
    }
    p.continuationSignature = `${rows.length}:${hash(rollout.policies.map(v => hashes.get(v)!).join("|"))}`;
  }
  if (resume) {
    const identity = (v: FittedEventValue) => JSON.stringify([v.contract, v.costs, v.targets, v.equities, v.prices, v.exposures,
      v.means, v.scales, v.penalty, v.samples, v.limits, v.trainingSignature, v.targetMode, v.pathHorizon, v.continuationSignature]);
    if (!resume.trainingSignature || identity(resume) !== identity(p) || !resume.tables.length || resume.tables.length > depths
      || resume.tables.some((t, i) => t.depth !== i + 1 || t.coefficients.length !== p.equities.length * p.prices.length * p.exposures.length
        || t.coefficients.some(b => b !== null && (b.length !== width + 1 || b.some(v => !Number.isFinite(v))))
        || JSON.stringify(t.correction?.options) !== JSON.stringify(boost))) throw new Error("Incompatible fitted-value checkpoint or training transitions");
    p.tables = [...resume.tables];
  }
  const x = rows.map(r => coordinates(p, r.features)), n = width + 1;
  const matrix = Array.from({ length: n }, () => new Float64Array(n));
  for (const row of x) for (let j = 0; j < n; j++) for (let k = 0; k <= j; k++) matrix[j][k] += row[j] * row[k] / rows.length;
  for (let j = 0; j < n; j++) {
    matrix[j][j] += j ? penalty : 0;
    for (let k = 0; k <= j; k++) {
      let value = matrix[j][k];
      for (let m = 0; m < k; m++) value -= matrix[j][m] * matrix[k][m];
      if (j === k) { if (!(value > 0)) throw new Error("Singular fitted-value design"); matrix[j][k] = Math.sqrt(value); }
      else matrix[j][k] = value / matrix[k][k];
    }
  }
  const solve = (rhs: Float64Array) => {
    const beta = [...rhs];
    for (let j = 0; j < n; j++) { for (let k = 0; k < j; k++) beta[j] -= matrix[j][k] * beta[k]; beta[j] /= matrix[j][j]; }
    for (let j = n - 1; j >= 0; j--) { for (let k = j + 1; k < n; k++) beta[j] -= matrix[k][j] * beta[k]; beta[j] /= matrix[j][j]; }
    return beta;
  };
  const cells = p.equities.length * p.prices.length * p.exposures.length;
  const held = rows.map(r => p.exposures.map(exposure => eventHolding(exposure, r.move, p.costs)));
  const settle = (exposure: number) => Math.log(1 - Math.abs(exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4);
  for (let depth = p.tables.length + 1; depth <= depths; depth++) {
    // At depth one the holding function is exactly independent of wealth and
    // price. Fit it once per exposure and share its coefficients across cells.
    const outputs = depth === 1 ? p.exposures.length : cells;
    const rhs = Array.from({ length: outputs }, () => new Float64Array(n)), valid = new Array<boolean>(outputs).fill(true);
    const targets: number[][] = [];
    for (let i = 0; i < rows.length; i++) {
      const evaluator = rollout?.policies?.[i] ?? p;
      const nextValues = depth > 1 && !rollout ? predictEventHoldingGrid(p, rows[i].nextFeatures, depth - 1) : undefined;
      const sampledValues = rollout && !rollout.minimumTurnover ? rollout.following[i].slice(0, depth - 1)
        .map((r, k) => predictEventHoldingGrid(evaluator, r.features, depth - k - 1)) : undefined;
      const target = new Array<number>(outputs);
      for (let output = 0; output < outputs; output++) {
        const xi = output % p.exposures.length, priceIndex = Math.floor(output / p.exposures.length) % p.prices.length;
        const wi = Math.floor(output / (p.exposures.length * p.prices.length)), h = held[i][xi];
        const nextAccount = { equity: p.equities[wi] * h.factor, price: p.prices[priceIndex] * (1 + rows[i].move.return), exposure: h.exposure };
        let continuation = h.liquidated ? -Infinity : depth === 1 ? settle(h.exposure) : rollout ? 0
          : chooseEventTrade(p, nextAccount, a => lookup(p, nextValues!, a, rows[i].nextLeaf)).value;
        if (rollout && depth > 1 && !h.liquidated) {
          let account = nextAccount;
          for (let k = 0; k < depth - 1; k++) {
            const step = rollout.following[i][k];
            // Choose the fixed hold/cap controller or earlier fitted action
            // before consuming this step's observed return. Never maximize it.
            const trade = rollout.minimumTurnover ? chooseEventTrade(p, account, () => 0)
              : chooseEventTrade(evaluator, account, a => lookup(evaluator, sampledValues![k], a, step.leaf));
            const nextHeld = eventHolding(trade.exposure, step.move, p.costs);
            if (nextHeld.liquidated || !Number.isFinite(trade.value)) { continuation = -Infinity; break; }
            continuation += Math.log(trade.equity / account.equity) + Math.log(nextHeld.factor);
            account = { equity: trade.equity * nextHeld.factor, price: trade.price * (1 + step.move.return), exposure: nextHeld.exposure };
          }
          if (Number.isFinite(continuation)) continuation += settle(account.exposure);
        }
        const value = target[output] = h.liquidated ? -Infinity : Math.log(h.factor) + continuation;
        if (!Number.isFinite(value)) { valid[output] = false; continue; }
        for (let f = 0; f < n; f++) rhs[output][f] += x[i][f] * value / rows.length;
      }
      targets.push(target);
    }
    const betas = rhs.map((r, output) => valid[output] ? solve(r) : null);
    const residual = boost ? targets.map((row, i) => row.map((value, output) => betas[output]
      ? value - betas[output]!.reduce((s, b, j) => s + b * x[i][j], 0) : 0)) : undefined;
    const correction = residual ? trainEventValueBoost(x, residual, boost!) : undefined;
    let squared = 0, count = 0;
    for (let i = 0; i < rows.length; i++) {
      const added = correction ? predictEventValueBoost(correction, x[i]) : undefined;
      for (let output = 0; output < outputs; output++) if (betas[output]) {
        squared += (betas[output]!.reduce((s, b, j) => s + b * x[i][j], 0) + (added?.[output] ?? 0) - targets[i][output]) ** 2; count++;
      }
    }
    if (!count) throw new Error("No finite fitted-value cells");
    p.tables.push({ depth, coefficients: depth === 1 ? Array.from({ length: cells }, (_, i) => betas[i % p.exposures.length]) : betas,
      ...(correction ? { correction } : {}), trainingMse: squared / count });
  }
  return p;
}
