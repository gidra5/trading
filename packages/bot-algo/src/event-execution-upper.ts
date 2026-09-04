import { cloneEventExecutionAtoms, type EventExecutionAtom } from "./event-execution-one-step.js";
import type { EventAccount } from "./event-log-policy.js";

/** Marked H1 upper relaxation: observe the first opening price before choosing
 * a continuous order, relax size limits and intrabar maintenance, retain fees
 * and exact path funding. Above-entry-cap holds/recovery are covered by allowing
 * any post-trade exposure up to max(entry cap, old opening exposure). A small
 * explicit cash credit covers inventory rounding before relaxing those caps.
 * This is an upper bound, never an executable order or a replacement forecast. */
export function prepareEventExecutionUpper(input: readonly EventExecutionAtom[]) {
  const atoms = cloneEventExecutionAtoms(input), c = atoms[0].path.costs;
  const f = (c.feeBps + c.slippageBps) / 10000, L = c.maxLeverage + 1e-8, m = c.maintenanceMargin;
  const pad = 1e-10, mass = atoms.reduce((s, a) => s + a.probability, 0);
  if (Math.abs(mass - 1) > 1e-12) throw new Error("Execution upper requires normalized probability within 1e-12");
  // Surviving holdings at the first open have |exposure| < 1/m. This also
  // keeps both fee budgets positive. Outside this supported contract return
  // an honest uninformative upper bound instead of extrapolating the formula.
  if (!(m > f && m > 0)) return Object.assign((_account: EventAccount) => Infinity,
    { groups: 0, supported: false, maxOptimizationGap: Infinity });
  const limit = 1 / m;
  const grouped = new Map<number, typeof atoms>();
  for (const atom of atoms) {
    const g = atom.path.openRatio, rows = grouped.get(g) ?? [];
    rows.push(atom); grouped.set(g, rows);
  }
  const groups = [...grouped].map(([open, outcomes]) => {
    const probability = outcomes.reduce((s, a) => s + a.probability, 0);
    const rows = outcomes.map(({ probability: p, path: a }) => ({ p: p / probability,
      close: a.closeRatio / open, long: a.longDebtGrowth, short: a.shortBorrowPriceIntegral / open }));
    const holding = (x: number) => {
      let value = 0;
      for (const a of rows) {
        const wealth = x < 0 ? 1 + x * (a.close - 1 + a.short)
          : x <= 1 ? 1 + x * (a.close - 1) : a.long + x * (a.close - a.long);
        if (!(wealth > 0)) return -Infinity;
        value += a.p * Math.log(wealth);
      }
      return value;
    };
    const target = (side: number) => {
      const d = side * f, transform = (x: number) => x / (1 + d * x);
      const cuts = [-limit, 0, 1, limit].filter(x => x >= -limit && x <= limit).map(transform);
      let best = -Infinity, bestTarget = 0, upper = -Infinity;
      for (let j = 1; j < cuts.length; j++) {
        let lo = cuts[j - 1], hi = cuts[j]; if (!(hi > lo)) continue;
        const middle = (lo + hi) / 2, x = middle / (1 - d * middle);
        const affine = rows.map(a => {
          const intercept = x > 1 ? a.long : 1;
          const slope = x < 0 ? a.close - 1 + a.short : x <= 1 ? a.close - 1 : a.close - a.long;
          return { p: a.p, a: intercept, b: slope - intercept * d };
        });
        for (const a of affine) {
          if (a.b > 0) lo = Math.max(lo, -a.a / a.b);
          else if (a.b < 0) hi = Math.min(hi, -a.a / a.b);
        }
        if (!(lo < hi)) continue;
        const score = (t: number) => {
          let value = 0, derivative = 0;
          for (const a of affine) {
            const wealth = a.a + a.b * t;
            if (!(wealth > 0)) return { value: -Infinity, derivative: a.b > 0 ? Infinity : -Infinity };
            value += a.p * Math.log(wealth); derivative += a.p * a.b / wealth;
          }
          return { value, derivative };
        };
        let left = lo, right = hi;
        for (let i = 0; i < 64; i++) {
          const t = left + (right - left) / 2; if (t === left || t === right) break;
          if (score(t).derivative > 0) left = t; else right = t;
        }
        let pieceUpper = Infinity;
        for (const t of [left, right, left + (right - left) / 2]) {
          const s = score(t); if (!Number.isFinite(s.value) || !Number.isFinite(s.derivative)) continue;
          pieceUpper = Math.min(pieceUpper, s.value + Math.max(s.derivative * (lo - t), s.derivative * (hi - t)));
          if (s.value > best) { best = s.value; bestTarget = t / (1 - d * t); }
        }
        upper = Math.max(upper, pieceUpper);
      }
      // Each signed-fee problem is concave in transformed target t, including
      // the funding kinks. Projecting an approximate maximizer onto a smaller
      // interval cannot increase its value regret beyond this global gap.
      return { exposure: bestTarget, gap: Math.max(0, upper - best) + pad };
    };
    return { open, probability, holding, buy: target(1), sell: target(-1) };
  });
  const query = (account: EventAccount) => {
    const { equity: E, price: P, exposure: x } = account;
    if (!(E > 0 && P > 0) || ![E, P, x].every(Number.isFinite)) throw new Error("Invalid execution upper account");
    const Q = x * E / P, rounded = Math.round(Q / c.quantityStep) * c.quantityStep;
    if (![Q, rounded].every(Number.isFinite)) return Infinity;
    let value = 0;
    for (const group of groups) {
      const G = P * group.open, openingEquity = E + Q * P * (group.open - 1);
      if (![G, openingEquity].every(Number.isFinite)) return Infinity;
      if (!(openingEquity > m * Math.abs(Q) * G)) return -Infinity;
      const before = Q * G / openingEquity, cap = Math.min(limit, Math.max(L, Math.abs(before)));
      const quantityError = Math.abs(Q - rounded) + 32 * Number.EPSILON
        * (Math.abs(Q) + (c.maxNotional + 1e-8) / G + c.quantityStep);
      // Actual cap/fees are checked BEFORE rounding inventory. In the
      // relaxation the order reaches that rounded inventory directly. This
      // credit covers the extra fee and supplies enough collateral that its
      // post-trade exposure stays within the original cap bound.
      const credit = (f + 1 / L) * quantityError * G;
      const budget = openingEquity + credit, exposure = Q * G / budget;
      const buy = Math.max(exposure, Math.min(cap, Math.max(-cap, group.buy.exposure)));
      const sell = Math.min(exposure, Math.min(cap, Math.max(-cap, group.sell.exposure)));
      const buyValue = Math.log1p(f * exposure) - Math.log1p(f * buy) + group.holding(buy) + group.buy.gap;
      const sellValue = Math.log1p(-f * exposure) - Math.log1p(-f * sell) + group.holding(sell) + group.sell.gap;
      const best = Math.max(group.holding(exposure), buyValue, sellValue);
      if (!Number.isFinite(best)) return Infinity;
      value += group.probability * (Math.log(budget / E) + best);
    }
    return value + pad;
  };
  return Object.assign(query, { groups: groups.length, supported: true,
    maxOptimizationGap: Math.max(...groups.flatMap(g => [g.buy.gap, g.sell.gap])) });
}
