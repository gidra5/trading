import type { MoveAtom } from "./event-distribution.js";
import type { EventCosts } from "./event-log-policy.js";
import { prepareEventHoldingLaw } from "./event-holding-law.js";

/** Upper relaxations for ordinary cap-feasible one-event orders. Ignore order
 * minima, lots and maintenance, retaining fees, borrowing and terminal
 * wealth. A shadow execution price m in [1-f,1+f] undercharges every real order.
 * With budget A=cash+m*quantity*price, its optimum is log(A)+K(m). Each such
 * function bounds the true continuation; its tangent is an affine upper bound.
 * A holding-wealth tangent additionally retains the maximum order notional.
 * Above-cap recovery exceptions must be bounded separately by the caller. */
export function eventOneStepUpper(kernel: readonly MoveAtom[], costs: EventCosts, terminal: "marked" | "friction") {
  const atoms = kernel.filter(a => a.probability > 0), f = (costs.feeBps + costs.slippageBps) / 10000, L = costs.maxLeverage + 1e-8;
  if (f * L >= 1) return () => null; // Cap tolerance crosses the shadow-budget singularity: use exhaustive bounds.
  const holdingLaw = atoms.length >= 64 && L <= 10 && f * L <= .5 ? prepareEventHoldingLaw(atoms, costs, terminal) : undefined;
  const rows = atoms.map(a => ({ p: a.probability, r: a.return, close: terminal === "friction" ? f * (1 + a.return) : 0,
    long: costs.longBorrowBpsPerDay / 10000 * a.duration / 1440, short: costs.shortBorrowBpsPerDay / 10000 * a.duration / 1440 }));
  const pad = 1e-11;
  const coefficient = (d: number) => {
    // t is final asset notional / shadow-price budget. On each borrowing/sign
    // piece, terminal wealth is affine in t and expected log is concave.
    const lower = -L / (1 - d * L), upper = L / (1 + d * L);
    const cuts = [lower, upper, 0, 1 / (1 + d)].filter(t => t >= lower && t <= upper).sort((a, b) => a - b);
    let best = -Infinity, exposure = 0;
    for (let j = 1; j < cuts.length; j++) {
      let lo = cuts[j - 1], hi = cuts[j]; if (lo === hi) continue;
      const mid = (lo + hi) / 2, borrowed = mid > 1 / (1 + d);
      const affine = rows.map(a => ({ p: a.p, a: 1 + (borrowed ? a.long : 0),
        b: a.r - d + (mid < 0 ? a.short : 0) - (borrowed ? a.long * (1 + d) : 0) - Math.sign(mid) * a.close }));
      for (const a of affine) {
        if (a.b > 0) lo = Math.max(lo, -a.a / a.b);
        if (a.b < 0) hi = Math.min(hi, -a.a / a.b);
      }
      if (!(lo < hi)) continue;
      const score = (t: number) => {
        let value = 0, derivative = 0;
        for (const a of affine) {
          const w = a.a + a.b * t;
          if (!(w > 0)) return { value: -Infinity, derivative: a.b > 0 ? Infinity : -Infinity };
          value += a.p * Math.log(w); derivative += a.p * a.b / w;
        }
        return { value, derivative };
      };
      let left = lo, right = hi;
      for (let iteration = 0; iteration < 64; iteration++) {
        const t = left + (right - left) / 2;
        if (t === left || t === right) break;
        if (score(t).derivative > 0) left = t; else right = t;
      }
      for (const t of [left, right, left + (right - left) / 2]) {
        const s = score(t); if (!Number.isFinite(s.value)) continue;
        // Tangent bound over the ORIGINAL feasible piece, so rounding the
        // stationary point cannot turn a lower estimate into an upper bound.
        const bound = s.value + Math.max(s.derivative * (lo - t), s.derivative * (hi - t));
        if (bound > best) { best = bound; exposure = t / (1 - d * t); }
      }
    }
    return { k: best + pad, exposure };
  };
  const edges = [...new Set([-f, f])].map(d => ({ d, ...coefficient(d) }));
  const noTradeLow = Math.min(...edges.map(a => a.exposure)) - 1e-10;
  const noTradeHigh = Math.max(...edges.map(a => a.exposure)) + 1e-10;
  const holdingTangent = (cash: number, quantity: number, price: number) => {
    const equity = cash + quantity * price, exposure = quantity * price / equity;
    let value = 0, dCash = 0, dQuantity = 0;
    if (!(equity > 0)) return null;
    if (holdingLaw) {
      // Absolute holding wealth is homogeneous: F = mass*log(E) + H(x).
      // The series' value/derivative remainders are <=1e-17. Roundoff remains
      // covered by the caller's padding. At zero choose a valid supergradient.
      const s = exposure === 0 ? { value: 0, derivative: holdingLaw.mean } : holdingLaw.score(exposure);
      if (!Number.isFinite(s.value) || !Number.isFinite(s.derivative)) return null;
      value = holdingLaw.mass * Math.log(equity) + s.value;
      dCash = (holdingLaw.mass - exposure * s.derivative) / equity;
      dQuantity = price * (holdingLaw.mass + (1 - exposure) * s.derivative) / equity;
    } else for (const a of rows) {
      const w = cash + quantity * price * (1 + a.r) - a.long * Math.max(0, -cash)
        - a.short * Math.max(0, -quantity) * price - a.close * Math.abs(quantity) * price;
      if (!(w > 0)) return null;
      value += a.p * Math.log(w);
      dCash += a.p * (1 + (cash < 0 ? a.long : 0)) / w;
      dQuantity += a.p * price * (1 + a.r + (quantity < 0 ? a.short : 0) - a.close * Math.sign(quantity)) / w;
    }
    return { value, dCash, dQuantity };
  };
  return (cash: number, quantity: number, price: number) => {
    const candidates: { d: number; k: number }[] = [...edges], equity = cash + quantity * price;
    const exposure = quantity * price / equity;
    // Outside the continuous no-trade region an endpoint shadow price already
    // supports the optimum. Avoid an O(atoms) marginal calculation per query.
    // Keeping only endpoint bounds remains valid even at a rounded boundary;
    // this shortcut can loosen a bound but cannot invalidate it.
    if (equity > 0 && exposure >= noTradeLow && exposure <= noTradeHigh) {
      const tangent = holdingTangent(cash, quantity, price);
      if (tangent) {
        const { value, dCash, dQuantity } = tangent;
        const d = dQuantity / (price * dCash) - 1, budget = cash + (1 + d) * quantity * price;
        if (d >= -f && d <= f && budget > 0) {
          // This holding portfolio is stationary under its marginal shadow
          // price. Concavity makes it a global relaxed optimum. The zero
          // position uses valid supergradients of borrowing/absolute costs.
          candidates.push({ d, k: value - Math.log(budget) + pad });
        }
      }
    }
    let best: { value: number; dCash: number; dQuantity: number } | null = null;
    for (const { d, k } of candidates) {
      const budget = cash + (1 + d) * quantity * price;
      if (!(budget > 0)) continue;
      const value = Math.log(budget) + k;
      if (!best || value < best.value) best = { value, dCash: 1 / budget, dQuantity: (1 + d) * price / budget };
    }
    const side = exposure < noTradeLow ? 1 : exposure > noTradeHigh ? -1 : 0;
    if (equity > 0 && side) {
      const target = side > 0 ? noTradeLow : noTradeHigh;
      const desiredNotional = (target * equity - quantity * price) / (1 + target * f * side);
      const maximum = costs.maxNotional + 1e-8;
      if (side * desiredNotional > maximum) {
        const order = side * maximum / price;
        const tangent = holdingTangent(cash - order * price - maximum * f, quantity + order, price);
        if (tangent && tangent.dCash > 0) {
          // F is concave in cash and quantity. For any incoming portfolio and
          // any order |u*P| <= M, its tangent changes by
          // (gQ-P*gC)*u - gC*f*P*|u|. Maximize this piecewise-linear expression
          // over [-M/P,M/P]. The resulting affine bound retains order capacity
          // at EVERY queried portfolio; it is not a pointwise clipping heuristic.
          const slope = tangent.dQuantity / price - tangent.dCash;
          const support = maximum * Math.max(0, slope - tangent.dCash * f, -slope - tangent.dCash * f);
          const value = tangent.value + support - side * maximum * slope + tangent.dCash * maximum * f + pad;
          if (!best || value < best.value) best = { ...tangent, value };
        }
      }
    }
    return best;
  };
}
