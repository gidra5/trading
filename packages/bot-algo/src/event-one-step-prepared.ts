import type { MoveAtom } from "./event-distribution.js";
import { eventCapAllowsTrade, eventHolding, eventTradeDecision, validateEventCosts,
  type EventAccount, type EventCosts, type EventTrade } from "./event-log-policy.js";
import { decideEventOneStep } from "./event-one-step.js";
import { prepareEventHoldingLaw } from "./event-holding-law.js";

/** Compile a fixed marked/friction law for many account queries. In each buy
 * or sell region, homogeneity makes the unconstrained optimal post-trade
 * exposure independent of equity and price. Order minima/maxima restrict
 * that concave region to intervals; only neighbors of the optimum and feasible
 * interval boundaries can win on the integer lattice. Recovery clips outside
 * the ordinary leverage cap are evaluated separately. */
export function prepareEventOneStep(kernel: readonly MoveAtom[], costs: EventCosts, terminal: "marked" | "friction") {
  validateEventCosts(costs);
  if (!kernel.length || !["marked", "friction"].includes(terminal)
    || kernel.some(a => ![a.probability, a.return, a.low, a.high, a.duration].every(Number.isFinite)
      || a.probability < 0 || a.return <= -1 || a.low <= -1 || a.low > Math.min(0, a.return)
      || a.high < Math.max(0, a.return) || a.duration <= 0)
    || Math.abs(kernel.reduce((s, a) => s + a.probability, 0) - 1) > 1e-8) throw new Error("Invalid prepared one-event law");
  const atoms = kernel.filter(a => a.probability > 0), f = (costs.feeBps + costs.slippageBps) / 10000;
  const holdingLaw = atoms.length >= 64 ? prepareEventHoldingLaw(atoms, costs, terminal) : undefined;
  const L = costs.maxLeverage + 1e-8;
  const rows = atoms.map(a => ({ ...a, long: costs.longBorrowBpsPerDay / 10000 * a.duration / 1440,
    short: costs.shortBorrowBpsPerDay / 10000 * a.duration / 1440, close: terminal === "friction" ? f * (1 + a.return) : 0 }));
  const targets: number[] = [];
  if (f * L < 1) for (const sign of [-1, 1]) {
    const d = sign * f, bottom = -L / (1 - d * L), top = L / (1 + d * L);
    const cuts = [...new Set([bottom, top, 0, 1 / (1 + d)].filter(t => t >= bottom && t <= top))].sort((a, b) => a - b);
    let best = -Infinity, target = 0;
    for (let i = 1; i < cuts.length; i++) {
      let lo = cuts[i - 1], hi = cuts[i];
      const mid = (lo + hi) / 2, side = Math.sign(mid), borrowed = mid > 1 / (1 + d);
      const affine = rows.map(a => {
        const b = -d + (mid < 0 ? a.short : 0) - (borrowed ? a.long * (1 + d) : 0);
        const base = 1 + (borrowed ? a.long : 0), worst = mid < 0 ? a.high : a.low;
        return { p: a.probability, a: base, b: b + a.return - side * a.close,
          survival: b + worst - costs.maintenanceMargin * side * (1 + worst) };
      });
      const positive = (a: number, b: number) => {
        if (b > 0) lo = Math.max(lo, -a / b);
        else if (b < 0) hi = Math.min(hi, -a / b);
        else if (!(a > 0)) hi = lo;
      };
      for (const a of affine) { positive(a.a, a.b); positive(a.a, a.survival); }
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
      for (let j = 0; j < 64; j++) {
        const t = left + (right - left) / 2; if (t === left || t === right) break;
        if (score(t).derivative > 0) left = t; else right = t;
      }
      // The supremum can be on a strict maintenance boundary. Keep it as a
      // candidate target, then validate neighboring integer orders directly.
      for (const t of [left, right, lo, hi]) {
        const value = score(t).value;
        if (value > best) { best = value; target = t / (1 - d * t); }
      }
    }
    targets.push(target);
  }
  const holdingValue = (exposure: number) => {
    if (holdingLaw) return holdingLaw.survives(exposure) ? holdingLaw.score(exposure).value : -Infinity;
    let value = 0;
    for (const a of atoms) {
      const held = eventHolding(exposure, a, costs);
      const wealth = held.factor - (terminal === "friction" ? f * Math.abs(exposure) * (1 + a.return) : 0);
      if (held.liquidated || !(wealth > 0)) return -Infinity;
      value += a.probability * Math.log(wealth);
    }
    return value;
  };
  type Result = ReturnType<typeof decideEventOneStep>;
  // Keep helpers outside the per-account path. Besides allocation, the TS
  // runtime names fresh closures with defineProperty on every invocation.
  // Bellman continuation can enter this path millions of times per action.
  const fallback = (account: EventAccount, valueOnly: boolean) => {
    const result = decideEventOneStep(atoms, account, costs, terminal);
    return valueOnly ? result.value : result;
  };
  const add = (candidates: number[], k: number) => { if (!candidates.includes(k)) candidates.push(k); };
  const ordinary = (k: number, E: number, P: number, N: number) => {
    const quantity = k * costs.quantityStep, turnover = Math.abs(quantity) * P, equity = E - turnover * f;
    return Math.abs(quantity) >= costs.minQuantity - 1e-12 && turnover >= costs.minNotional - 1e-8
      && turnover <= costs.maxNotional + 1e-8 && equity > 0 && Math.abs((N + quantity * P) / equity) <= L;
  };
  function solve(account: EventAccount, valueOnly: true): number;
  function solve(account: EventAccount, valueOnly: false): Result;
  function solve(account: EventAccount, valueOnly: boolean): Result | number {
    const { equity: E, price: P, exposure: x } = account, step = costs.quantityStep, unit = P * step, N = x * E;
    const maximum = Math.floor(costs.maxNotional / unit + 1e-8);
    // The general solver retains its stable discrete-derivative search near
    // singular costs or a lattice too fine for reliable target inversion.
    if (!(E > 0 && P > 0) || !Number.isFinite(E) || !Number.isFinite(P) || !Number.isFinite(x) || !Number.isSafeInteger(maximum)
      || maximum > 1e12 || f * Math.max(L, Math.abs(x)) >= .99) return fallback(account, valueOnly);
    // A surviving account between the directional optima is already globally
    // optimal before imposing order constraints. Check this before allocating
    // candidate sets; Bellman recursion visits this region millions of times.
    const holdOptimal = targets.length === 2 && (x > targets[1] + 1e-12 && x < targets[0] - 1e-12
      || x === targets[0] && x === targets[1]);
    if (holdOptimal && Math.abs(x) <= costs.maxLeverage + 1e-9) {
      const score = holdingValue(x);
      if (Number.isFinite(score)) return valueOnly ? score : {
        ...eventTradeDecision(account, { ...account, quantity: 0, turnover: 0, cost: 0 }, score), feasible: true,
        search: { evaluatedOrders: 1, searchedIntervals: 0, derivativeEvaluations: 0 },
      };
    }
    const candidates = [0];
    const minimum = Math.max((costs.minNotional - 1e-8) / unit, (costs.minQuantity - 1e-12) / step);
    // For each order direction, ordinary cap/min/max constraints intersect in
    // ONE integer interval. Its closest neighbors of the continuous optimum
    // suffice by concavity. Validate outward-rounded endpoints using the real
    // execution arithmetic instead of constructing every boundary candidate.
    for (const [i, sign] of [-1, 1].entries()) {
      const minLots = Math.max(1, Math.ceil(minimum) - 2);
      let low = Math.max(sign > 0 ? minLots : -maximum,
        Math.ceil((-L * E - N) / (unit * (1 - L * f * sign))) - 2);
      let high = Math.min(sign > 0 ? maximum : -minLots,
        Math.floor((L * E - N) / (unit * (1 + L * f * sign))) + 2);
      let adjustments = 0;
      while (low <= high && !ordinary(low, E, P, N) && adjustments++ < 6) low++;
      while (low <= high && !ordinary(high, E, P, N) && adjustments++ < 12) high--;
      if (low > high) continue;
      if (!ordinary(low, E, P, N) || !ordinary(high, E, P, N)) return fallback(account, valueOnly);
      const target = targets[i], optimum = (target * E - N) / (unit * (1 + target * f * sign));
      if (!Number.isFinite(optimum)) return fallback(account, valueOnly);
      add(candidates, Math.max(low, Math.min(high, Math.floor(optimum))));
      add(candidates, Math.max(low, Math.min(high, Math.ceil(optimum))));
    }
    if (Math.abs(x) > costs.maxLeverage) {
      const firstClip = Math.max(1, Math.ceil((costs.maxNotional - unit - 1e-8) / unit));
      if (maximum - firstClip > 4) return fallback(account, valueOnly);
      for (let k = firstClip; k <= maximum; k++) { add(candidates, k); add(candidates, -k); }
    }
    let bestQuantity = 0, bestEquity = E, bestExposure = x, bestTurnover = 0, bestCost = 0;
    let value = -Infinity, feasible = false, evaluatedOrders = 0;
    for (const k of candidates) {
      const quantity = k * step, turnover = Math.abs(quantity) * P, cost = turnover * f, equity = E - cost;
      if (!k ? Math.abs(x) > costs.maxLeverage + 1e-9 : Math.abs(quantity) < costs.minQuantity - 1e-12
        || turnover < costs.minNotional - 1e-8 || turnover > costs.maxNotional + 1e-8 || equity <= 0) continue;
      const exposure = k ? (N + quantity * P) / equity : x;
      if (k && !eventCapAllowsTrade(x, exposure, turnover, P, costs)) continue;
      evaluatedOrders++;
      const score = Math.log(equity / E) + holdingValue(exposure);
      if (!feasible || score > value + 1e-14 || score === value && (Math.abs(quantity) < Math.abs(bestQuantity)
        || !Number.isFinite(value) && Math.abs(exposure) < Math.abs(bestExposure))) {
        bestQuantity = quantity; bestEquity = equity; bestExposure = exposure; bestTurnover = turnover; bestCost = cost;
        value = score; feasible = true;
      }
    }
    if (valueOnly) return value;
    const best: EventTrade = { equity: bestEquity, price: P, exposure: bestExposure,
      quantity: bestQuantity, turnover: bestTurnover, cost: bestCost };
    return { ...eventTradeDecision(account, best, value), feasible,
      search: { evaluatedOrders, searchedIntervals: 0, derivativeEvaluations: 0 } };
  }
  // Recursive callers need the identical feasible value without allocating
  // an order/signal result that is immediately discarded.
  return Object.assign((account: EventAccount) => solve(account, false), {
    value: (account: EventAccount) => solve(account, true),
  });
}
