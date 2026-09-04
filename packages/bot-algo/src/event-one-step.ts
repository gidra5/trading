import type { MoveAtom } from "./event-distribution.js";
import { eventCapAllowsTrade, eventHolding, eventTradeDecision, validateEventCosts,
  type EventAccount, type EventCosts, type EventTrade } from "./event-log-policy.js";

export type EventTerminal = "marked" | "friction" | "market";

/** Globally search the feasible one-event ORDER lattice under a fixed kernel.
 * Within each fee/position/borrowing/terminal-dust region, terminal wealth is
 * affine in the order lot count. Its expected log is concave; binary search
 * the monotone discrete derivative. Check region boundaries explicitly.
 * The supplied event law starts at the decision price, with no opening gap.
 * This solves a finite one-event problem, not an arbitrary fitted critic. */
export function decideEventOneStep(kernel: readonly MoveAtom[], account: EventAccount, costs: EventCosts,
  terminal: EventTerminal) {
  validateEventCosts(costs);
  if (!kernel.length || !["marked", "friction", "market"].includes(terminal) || !(account.equity > 0 && account.price > 0)
    || !Object.values(account).every(Number.isFinite)
    || kernel.some(a => ![a.probability, a.return, a.low, a.high, a.duration].every(Number.isFinite)
      || a.probability < 0 || a.return <= -1 || a.low <= -1 || a.low > Math.min(0, a.return)
      || a.high < Math.max(0, a.return) || a.duration <= 0)
    || Math.abs(kernel.reduce((s, a) => s + a.probability, 0) - 1) > 1e-8) throw new Error("Invalid exact one-event problem");
  const atoms = kernel.filter(a => a.probability > 0), { equity: E, price: P, exposure: x } = account;
  const f = (costs.feeBps + costs.slippageBps) / 10000, step = costs.quantityStep, unit = P * step, N = x * E;
  const maximum = Math.floor(costs.maxNotional / unit + 1e-8);
  if (!Number.isSafeInteger(maximum)) throw new Error("One-event order lattice exceeds integer precision");
  const stats = { evaluatedOrders: 0, searchedIntervals: 0, derivativeEvaluations: 0 };
  if (terminal !== "market" && Math.abs(x) <= costs.maxLeverage + 1e-9) {
    // Marked/friction terminal wealth is concave in order notional: fees,
    // borrowing and liquidation friction are convex deductions. If both
    // one-sided derivatives bracket zero, holding is globally optimal even
    // before imposing lot/minimum constraints. This avoids repeated lattice
    // searches for quiet successor states in deeper Bellman expectations.
    let score = 0, left = 0, right = 0, survives = true;
    for (const atom of atoms) {
      const held = eventHolding(x, atom, costs), close = terminal === "friction" ? f * (1 + atom.return) : 0;
      const wealth = held.factor - close * Math.abs(x);
      if (held.liquidated || !(wealth > 0)) { survives = false; break; }
      const long = costs.longBorrowBpsPerDay / 10000 * atom.duration / 1440;
      const short = costs.shortBorrowBpsPerDay / 10000 * atom.duration / 1440;
      const dl = f + atom.return + (x <= 0 ? short : 0) - (x > 1 ? long * (1 - f) : 0) - (x > 0 ? close : -close);
      const dr = -f + atom.return + (x < 0 ? short : 0) - (x >= 1 ? long * (1 + f) : 0) - (x >= 0 ? close : -close);
      score += atom.probability * Math.log(wealth);
      left += atom.probability * dl / wealth; right += atom.probability * dr / wealth;
    }
    if (survives && left > 1e-12 && right < -1e-12) {
      stats.evaluatedOrders = 1; stats.derivativeEvaluations = 2;
      return { ...eventTradeDecision(account, { ...account, quantity: 0, turnover: 0, cost: 0 }, score),
        feasible: true, search: stats };
    }
  }
  const visited = new Set<number>(), guards = new Set<number>([-maximum, maximum, 0]);
  let best: EventTrade = { ...account, quantity: 0, turnover: 0, cost: 0 }, value = -Infinity, feasible = false;
  const trade = (k: number): EventTrade | null => {
    if (!k) return Math.abs(x) <= costs.maxLeverage + 1e-9 ? { ...account, quantity: 0, turnover: 0, cost: 0 } : null;
    const quantity = k * step, turnover = Math.abs(quantity) * P, cost = turnover * f, equity = E - cost;
    if (Math.abs(quantity) < costs.minQuantity - 1e-12 || turnover < costs.minNotional - 1e-8
      || turnover > costs.maxNotional + 1e-8 || equity <= 0) return null;
    const exposure = (N + quantity * P) / equity;
    return eventCapAllowsTrade(x, exposure, turnover, P, costs) ? { equity, price: P, exposure, quantity, turnover, cost } : null;
  };
  const consider = (k: number) => {
    if (!Number.isSafeInteger(k) || Math.abs(k) > maximum || visited.has(k)) return;
    visited.add(k); stats.evaluatedOrders++;
    const next = trade(k);
    if (!next) return;
    let score = Math.log(next.equity / E);
    for (const atom of atoms) {
      const held = eventHolding(next.exposure, atom, costs);
      if (held.liquidated) { score = -Infinity; break; }
      const quantity = Math.abs(next.exposure * next.equity / P), notional = quantity * P * (1 + atom.return);
      const close = terminal === "friction" || (terminal === "market" && quantity >= costs.minQuantity - 1e-12 && notional >= costs.minNotional - 1e-8);
      const wealth = held.factor - (close ? notional * f / next.equity : 0);
      if (!(wealth > 0)) { score = -Infinity; break; }
      score += atom.probability * Math.log(wealth);
    }
    if (!feasible || score > value + 1e-14 || (score === value && (Math.abs(next.quantity) < Math.abs(best.quantity)
      || (!Number.isFinite(value) && Math.abs(next.exposure) < Math.abs(best.exposure))))) {
      best = next; value = score; feasible = true;
    }
  };
  // All integer neighbors of a real branch boundary are evaluated directly;
  // open integer intervals between them cannot cross an economic kink.
  const boundary = (k: number) => {
    if (!Number.isFinite(k) || k < -maximum - 2 || k > maximum + 2) return;
    for (const n of [Math.floor(k) - 1, Math.floor(k), Math.ceil(k), Math.ceil(k) + 1])
      if (Math.abs(n) <= maximum) guards.add(n);
  };
  boundary(-N / unit);
  const minimum = Math.max((costs.minQuantity - 1e-12) / step, (costs.minNotional - 1e-8) / unit);
  boundary(minimum); boundary(-minimum);
  for (const side of [-1, 1]) {
    boundary(E / (f * unit * side)); // positive post-order equity
    boundary((E - N) / (unit * (1 + f * side))); // long borrowing starts
    for (const cap of [costs.maxLeverage + 1e-8, -costs.maxLeverage - 1e-8,
      Math.abs(x) - 1e-9, -Math.abs(x) + 1e-9])
      boundary((cap * E - N) / (unit * (1 + cap * f * side)));
    boundary(side * (costs.maxNotional - unit - 1e-8) / unit);
  }
  if (terminal === "market") for (const atom of atoms) {
    const threshold = Math.max(0, (costs.minQuantity - 1e-12) * P, (costs.minNotional - 1e-8) / (1 + atom.return));
    boundary((threshold - N) / unit); boundary((-threshold - N) / unit);
  }
  // Above-cap recovery may admit the final one or two maximum-notional lots
  // even before exposure returns below the cap. Check those isolated actions.
  for (const side of [-1, 1]) for (let i = 0; i <= 2; i++) boundary(side * (maximum - i));
  const points = [...guards].sort((a, b) => a - b);
  for (const k of points) consider(k);
  for (let index = 1; index < points.length; index++) {
    let low = points[index - 1] + 1, high = points[index] - 1;
    if (low > high) continue;
    const middle = (low + high) / 2, sign = Math.sign(middle), e1 = -f * unit * sign;
    const e = E + e1 * middle, n = N + unit * middle;
    // Root order feasibility is constant inside this region, apart from
    // outcome-dependent liquidation constraints handled below.
    if (!trade(Math.floor(middle))) continue;
    stats.searchedIntervals++;
    const positionSign = Math.sign(n), affine: Array<{ a: number; b: number; probability: number }> = [];
    const positive = (a: number, b: number) => {
      if (!b) { if (!(a > 0)) high = low - 1; return; }
      const crossing = -a / b;
      if (crossing >= low - 1 && crossing <= high + 1)
        for (const k of [Math.floor(crossing) - 1, Math.floor(crossing), Math.ceil(crossing), Math.ceil(crossing) + 1]) consider(k);
      if (b > 0) low = Math.max(low, Math.floor(crossing) + 1);
      else high = Math.min(high, Math.ceil(crossing) - 1);
    };
    for (const atom of atoms) {
      const rate = (n < 0 ? costs.shortBorrowBpsPerDay : costs.longBorrowBpsPerDay) / 10000 * atom.duration / 1440;
      const debt0 = n < 0 ? -N : n > e ? N - E : 0;
      const debt1 = n < 0 ? -unit : n > e ? unit - e1 : 0;
      const end0 = E + N * atom.return - debt0 * rate, end1 = e1 + unit * atom.return - debt1 * rate;
      const worst = n >= 0 ? atom.low : atom.high;
      positive(E + N * worst - debt0 * rate - costs.maintenanceMargin * positionSign * N * (1 + worst),
        e1 + unit * worst - debt1 * rate - costs.maintenanceMargin * positionSign * unit * (1 + worst));
      positive(end0, end1);
      const close = terminal === "friction" || (terminal === "market" && Math.abs(n) / P >= costs.minQuantity - 1e-12
        && Math.abs(n) * (1 + atom.return) >= costs.minNotional - 1e-8);
      const a = end0 - (close ? f * positionSign * N * (1 + atom.return) : 0);
      const b = end1 - (close ? f * positionSign * unit * (1 + atom.return) : 0);
      positive(a, b); affine.push({ a, b, probability: atom.probability });
      if (low > high) break;
    }
    if (low > high) continue;
    consider(low); consider(high);
    // F(k+1)-F(k) is non-increasing for a sum of log-affine wealth terms.
    while (low < high) {
      const k = Math.floor((low + high) / 2); let delta = 0;
      stats.derivativeEvaluations++;
      for (const row of affine) delta += row.probability * Math.log1p(row.b / (row.a + row.b * k));
      if (delta > 0) low = k + 1; else high = k;
    }
    for (let k = low - 1; k <= low + 1; k++) consider(k);
  }
  return { ...eventTradeDecision(account, best, value), feasible, search: stats };
}
