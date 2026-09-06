import { cloneEventExecutionAtoms, type EventExecutionAtom } from "./event-execution-one-step.js";
import type { EventAccount } from "./event-log-policy.js";

/**
 * Convex upper bound on absolute expected marked terminal equity after one
 * committed request. The request remains common across outcomes. Order-size,
 * cap, maintenance and availability checks are relaxed by allowing each path
 * to keep the position or accept the request in its own favor. Funding is
 * bounded by the best of every possible cash/inventory regime. The request
 * objective is convex on each side of zero, so only zero and the two largest
 * requests need evaluation.
 *
 * The result is deliberately an arithmetic-wealth bound. It becomes an upper
 * bound on expected log wealth only when Jensen's inequality is applied over
 * the complete remaining outcome tree by the caller.
 */
export function prepareEventExecutionRiskNeutralUpper(input: readonly EventExecutionAtom[], options: { requestBins?: number } = {}) {
  const atoms = cloneEventExecutionAtoms(input), c = atoms[0].path.costs;
  if (Math.abs(atoms.reduce((sum, atom) => sum + atom.probability, 0) - 1) > 1e-12)
    throw new Error("Execution risk-neutral upper requires normalized probability within 1e-12");
  const fee = (c.feeBps + c.slippageBps) / 10000;
  const requestBins = options.requestBins ?? 1;
  if (!Number.isInteger(requestBins) || requestBins < 1 || requestBins > 129) throw new Error("Invalid execution request-bin count");
  const minimumOpen = Math.min(...atoms.filter(atom => atom.path.openingAvailable).map(atom => atom.path.openRatio));
  const grouped = new Map<number, typeof atoms>();
  for (const atom of atoms) {
    const rows = grouped.get(atom.path.openRatio) ?? [];
    rows.push(atom); grouped.set(atom.path.openRatio, rows);
  }
  const openings = [...grouped].sort((a, b) => a[0] - b[0]);
  return (account: EventAccount) => {
    const { equity: E, price: P, exposure: x } = account;
    if (!(E > 0 && P > 0) || ![E, P, x].every(Number.isFinite)) throw new Error("Invalid execution upper account");
    const Q = x * E / P, C = E - Q * P;
    if (![Q, C].every(Number.isFinite)) return Infinity;
    const maximum = Number.isFinite(minimumOpen) ? (c.maxNotional + 1e-8) / (P * minimumOpen) : 0;
    if (!Number.isFinite(maximum)) return Infinity;
    const qScale = Math.max(1, Math.abs(Q), maximum, c.quantityStep), rounding = c.quantityStep / 2
      + 64 * Number.EPSILON * qScale;
    const value = (request: number, possibleOpen: number) => {
      const choices = openings.map(([open, rows]) => {
        let accepted = 0, rejected = 0;
        for (const atom of rows) {
          const p = atom.path, G = P * open, close = P * p.closeRatio;
          // Ignoring both debt charges is an affine upper on marked wealth.
          // Rounding changes cash and inventory in opposite directions, so its
          // largest possible contribution is |close-open| times half a lot.
          const hold = Math.max(0, C + Q * close), slope = close - G;
          const trade = Math.max(0, C + Q * close + request * slope
            - Math.abs(request) * G * fee + rounding * Math.abs(slope));
          rejected += atom.probability * hold;
          accepted += atom.probability * (p.openingAvailable && open <= possibleOpen ? trade : hold);
        }
        return { accepted, rejected };
      });
      // For a fixed request, ordinary acceptance is an interval in ascending
      // opening price. The narrow maximum-order recovery exception can add one
      // more interval. Maximizing over every union of up to two intervals is a
      // convex relaxation independent of the incoming balance.
      const states = Array.from({ length: 3 }, () => [-Infinity, -Infinity]);
      states[0][0] = 0;
      for (const row of choices) {
        const next = Array.from({ length: 3 }, () => [-Infinity, -Infinity]);
        for (let used = 0; used <= 2; used++) for (let inside = 0; inside <= 1; inside++) {
          const prior = states[used][inside]; if (prior === -Infinity) continue;
          next[used][0] = Math.max(next[used][0], prior + row.rejected);
          if (inside) next[used][1] = Math.max(next[used][1], prior + row.accepted);
          else if (used < 2) next[used + 1][1] = Math.max(next[used + 1][1], prior + row.accepted);
        }
        for (let used = 0; used <= 2; used++) for (let inside = 0; inside <= 1; inside++) states[used][inside] = next[used][inside];
      }
      return Math.max(...states.flat());
    };
    const thresholds = openings.filter(([, rows]) => rows.some(atom => atom.path.openingAvailable))
      .map(([open]) => (c.maxNotional + 1e-8) / (P * open)).sort((a, b) => a - b);
    const cuts = [0];
    for (let i = 1; i < requestBins; i++) cuts.push(thresholds[Math.floor(i * (thresholds.length - 1) / requestBins)]);
    cuts.push(maximum);
    const unique = [...new Set(cuts)].sort((a, b) => a - b);
    let upper = value(0, Infinity);
    for (let i = 1; i < unique.length; i++) {
      const low = unique[i - 1], high = unique[i];
      // Every request in this band can satisfy the maximum-size check only at
      // openings no larger than M/low. Hold the relaxed eligibility set fixed
      // across the band; the remaining objective is a maximum of affine
      // functions and therefore reaches its maximum at a band endpoint.
      const possibleOpen = low ? (c.maxNotional + 1e-8) / (P * low) : Infinity;
      for (const request of [-high, -low, low, high]) upper = Math.max(upper, value(request, possibleOpen));
    }
    return Number.isFinite(upper) ? upper + 1e-10 * Math.max(1, Math.abs(upper)) : Infinity;
  };
}
